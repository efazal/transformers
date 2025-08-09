"""
JIT (Just-In-Time) Checkpointing for Transformers Trainer

This module provides functionality for asynchronous checkpointing upon receiving
termination signals, enabling graceful shutdown in preemptible environments.
"""

import os
import signal
import threading
import time
from datetime import timedelta
from typing import Optional, Any

import torch

from .trainer_callback import TrainerCallback
from .utils import logging


logger = logging.get_logger(__name__)


class JITCheckpointManager:
    def __init__(self, trainer, grace_period: float = 30.0):
        self.trainer = trainer
        self.grace_period = grace_period
        self.checkpoint_requested = False
        self.checkpoint_in_progress = False
        self.checkpoint_thread = None
        self.checkpoint_stream = None
        self._original_sigterm_handler = None

        if torch.cuda.is_available():
            self.checkpoint_stream = torch.cuda.Stream()

    def setup_signal_handler(self):
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def cleanup_signal_handler(self):
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            self._original_sigterm_handler = None

    def _sigterm_handler(self, signum, frame):
        if self.checkpoint_requested:
            return

        logger.info(f"SIGTERM received, initiating JIT checkpoint with {self.grace_period}s grace period")
        self.checkpoint_requested = True

        # In Kubernetes/PyTorchJob, ALL pods receive SIGTERM simultaneously
        # We need to handle this carefully for distributed training
        # if torch.distributed.is_initialized():
        #     rank = torch.distributed.get_rank()
        #     world_size = torch.distributed.get_world_size()
        #     logger.info(f"Distributed training detected: rank={rank}, world_size={world_size}")
        #
        #     if rank == 0:
        #         # Master rank: Try to checkpoint immediately
        #         logger.info("Master rank received SIGTERM, attempting immediate checkpoint")
        #         self.checkpoint_thread = threading.Thread(
        #             target=self._kubernetes_checkpoint_strategy,
        #             daemon=True
        #         )
        #         self.checkpoint_thread.start()
        #     else:
        #         # Worker ranks: Try to stay alive a bit longer to help with checkpoint
        #         logger.info(f"Worker rank {rank} received SIGTERM, will try to assist with checkpoint")
        #         self.checkpoint_thread = threading.Thread(
        #             target=self._worker_assist_strategy,
        #             daemon=True
        #         )
        #         self.checkpoint_thread.start()
        # else:
        #     # Single node training
        #     self.checkpoint_thread = threading.Thread(
        #         target=self._async_checkpoint_with_timeout,
        #         daemon=True
        #     )
        #     self.checkpoint_thread.start()

    def _kubernetes_checkpoint_strategy(self):
        """
        Special checkpoint strategy for Kubernetes where all pods get SIGTERM simultaneously.
        We need to checkpoint as fast as possible before worker nodes disconnect.
        """
        try:
            logger.info("Starting Kubernetes-aware JIT checkpoint")
            # self._save_kubernetes_checkpoint()
            self._async_checkpoint_with_timeout()
        except Exception as e:
            logger.error(f"Error in Kubernetes checkpoint strategy: {e}")
        finally:
            if hasattr(self.trainer, 'control'):
                self.trainer.control.should_training_stop = True

    def _worker_assist_strategy(self):
        """
        Strategy for worker nodes to assist with checkpoint before shutting down.
        Workers will try to stay responsive for a short time to help master checkpoint.
        """
        try:
            logger.info("Worker node assisting with checkpoint...")

        except Exception as e:
            logger.error(f"Error in worker assist strategy: {e}")
        finally:
            if hasattr(self.trainer, 'control'):
                self.trainer.control.should_training_stop = True

    def _save_kubernetes_checkpoint(self):
        """Save checkpoint optimized for Kubernetes distributed environment."""
        try:
            original_step = self.trainer.state.global_step
            logger.info(f"Saving Kubernetes JIT checkpoint at step {original_step}")

            # Use the trainer's checkpoint method but with minimal distributed coordination
            if self.checkpoint_stream is not None:
                with torch.cuda.stream(self.checkpoint_stream):
                    logger.info("Checkpointing with stream synchronization...")
                    self.trainer._save_checkpoint(self.trainer.model, trial=None)
                self.checkpoint_stream.synchronize()
            else:
                logger.info("Checkpointing without stream synchronization...")
                self.trainer._save_checkpoint(self.trainer.model, trial=None)

            # Rename checkpoint if we're the main process
            if self.trainer.is_world_process_zero():
                run_dir = self.trainer._get_output_dir(trial=None)
                regular_checkpoint_dir = os.path.join(run_dir, f"checkpoint-{original_step}")
                jit_checkpoint_dir = os.path.join(run_dir, f"checkpoint-jit-{original_step}")

                if os.path.exists(regular_checkpoint_dir):
                    os.rename(regular_checkpoint_dir, jit_checkpoint_dir)
                    logger.info(f"Kubernetes JIT checkpoint saved to {jit_checkpoint_dir}")
                else:
                    logger.warning(f"Expected checkpoint directory not found: {regular_checkpoint_dir}")

        except Exception as e:
            logger.error(f"Failed to save Kubernetes checkpoint: {e}")
            raise

    def _async_checkpoint_with_timeout(self):
        """Original checkpoint method for non-Kubernetes environments."""
        start_time = time.time()

        try:
            self._execute_jit_checkpoint()
        except Exception as e:
            logger.error(f"Error during JIT checkpoint: {e}")
        finally:
            if hasattr(self.trainer, 'control'):
                self.trainer.control.should_training_stop = True

    def _execute_jit_checkpoint(self):
        if self.checkpoint_in_progress:
            logger.warning("Checkpoint already in progress, skipping")
            return

        self.checkpoint_in_progress = True

        try:
            logger.info("Starting JIT checkpoint save")

            # For distributed training, we need to be more careful about NCCL operations
            # First, ensure all processes are synchronized before starting checkpoint
            if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator.num_processes > 1:
                try:
                    # Give a shorter timeout for the initial sync
                    logger.info("Synchronizing distributed processes before JIT checkpoint")
                    torch.distributed.barrier()
                except Exception as e:
                    logger.warning(f"Failed to synchronize processes before checkpoint, continuing: {e}")

            # if self.checkpoint_stream is not None:
            #     with torch.cuda.stream(self.checkpoint_stream):
            #         self._save_jit_checkpoint()
            #     self.checkpoint_stream.synchronize()
            # else:
            #     self._save_jit_checkpoint()
            self._save_jit_checkpoint()

            logger.info("JIT checkpoint completed successfully")

        except Exception as e:
            logger.error(f"Failed to complete JIT checkpoint: {e}")
            raise
        finally:
            self.checkpoint_in_progress = False

    def _save_jit_checkpoint(self):
        try:
            original_step = self.trainer.state.global_step
            logger.info(f"Saving JIT checkpoint at step {original_step}")

            # Call the trainer's checkpoint method directly without timeout manipulation
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

            if self.trainer.is_world_process_zero():
                run_dir = self.trainer._get_output_dir(trial=None)
                regular_checkpoint_dir = os.path.join(run_dir, f"checkpoint-{original_step}")
                jit_checkpoint_dir = os.path.join(run_dir, f"checkpoint-jit-{original_step}")

                if os.path.exists(regular_checkpoint_dir):
                    os.rename(regular_checkpoint_dir, jit_checkpoint_dir)
                    logger.info(f"JIT checkpoint saved to {jit_checkpoint_dir}")
                else:
                    logger.warning(f"Expected checkpoint directory not found: {regular_checkpoint_dir}")

        except Exception as e:
            logger.error(f"Failed to save JIT checkpoint: {e}")
            raise

    def should_checkpoint_now(self) -> bool:
        return self.checkpoint_requested

    def cleanup(self):
        self.cleanup_signal_handler()

        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            self.checkpoint_thread.join(timeout=5.0)

        if self.checkpoint_stream is not None:
            self.checkpoint_stream.synchronize()


class JITCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.jit_manager: Optional[JITCheckpointManager] = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        if trainer.args.jit_checkpoint_on_sigterm:
            self.jit_manager = JITCheckpointManager(
                trainer=trainer,
                grace_period=trainer.args.jit_checkpoint_grace_period
            )
            self.jit_manager.setup_signal_handler()
            logger.info("JIT checkpointing enabled for Kubernetes/PyTorchJob environment")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            logger.info("SIGTERM detected, triggering JIT checkpoint before optimizer step")
            try:
                # Do checkpoint SYNCHRONOUSLY - don't use daemon thread
                self.jit_manager._execute_jit_checkpoint()
                self.jit_manager.checkpoint_requested = False  # Mark as completed
                logger.info("JIT checkpoint completed at pre-optimizer step")
            except Exception as e:
                logger.error(f"Error in pre-optimizer step checkpoint: {e}")
            finally:
                control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        # Log
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            logger.info("SIGTERM detected, triggering JIT checkpoint on step end")
            try:
                # Do checkpoint SYNCHRONOUSLY - don't use daemon thread
                self.jit_manager._execute_jit_checkpoint()
                self.jit_manager.checkpoint_requested = False  # Mark as completed
                logger.info("JIT checkpoint completed at step end")
            except Exception as e:
                logger.error(f"Error in step end checkpoint strategy: {e}")
            finally:
                control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.jit_manager:
            self.jit_manager.cleanup()
            logger.info("JIT checkpoint manager cleaned up")
