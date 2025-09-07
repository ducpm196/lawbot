#!/usr/bin/env python3
"""
Training Engine - LawBot v8.0
=============================

Core training engine for all LawBot models.
Optimized, simple, efficient, consistent, and maintainable.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from core.utils.logging_manager import get_logger

logger = get_logger(__name__)


class TrainingEngine:
    """Training engine for LawBot models."""

    def __init__(
        self,
        model_name: str,
        train_dataset,
        eval_dataset,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        train_batch_size: int = 16,
        max_length: int = 256,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize training engine."""
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.scaler = GradScaler() if self.use_mixed_precision else None

        logger.info(f"Training engine initialized with device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def setup_model_and_tokenizer(self, num_labels: int = 2):
        """Setup model and tokenizer."""
        try:
            logger.info(f"Loading model and tokenizer: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )

            # Move to device
            self.model.to(self.device)

            logger.info(f"Model loaded successfully: {self.model_name}")
            logger.info(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        except Exception as e:
            logger.error(f"Failed to setup model and tokenizer: {e}")
            raise

    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        try:
            # Setup optimizer
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                eps=1e-8,
            )

            # Setup scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * num_training_steps),
                num_training_steps=num_training_steps,
            )

            logger.info("Optimizer and scheduler setup completed")

        except Exception as e:
            logger.error(f"Failed to setup optimizer and scheduler: {e}")
            raise

    def setup_dataloaders(self):
        """Setup training and evaluation dataloaders."""
        try:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                pin_memory=True,
            )

            if self.eval_dataset:
                self.eval_dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=False,
                    pin_memory=True,
                )

            logger.info(
                f"DataLoaders setup completed. Train batches: {len(self.train_dataloader)}"
            )

        except Exception as e:
            logger.error(f"Failed to setup dataloaders: {e}")
            raise

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        logger.info(f"Starting epoch {epoch + 1}/{self.num_train_epochs}")

        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                # Log progress
                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"train_loss": avg_loss}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if not self.eval_dataloader:
            logger.warning("No evaluation dataset available")
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        logger.info("Starting evaluation...")

        with torch.no_grad():
            for batch in self.eval_dataloader:
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Evaluation completed. Average loss: {avg_loss:.4f}")

        return {"eval_loss": avg_loss}

    def train(self) -> Dict[str, Any]:
        """Complete training process."""
        try:
            logger.info("🚀 Starting training process...")
            start_time = time.time()

            # Setup components
            num_training_steps = len(self.train_dataloader) * self.num_train_epochs
            self.setup_model_and_tokenizer()
            self.setup_optimizer_and_scheduler(num_training_steps)
            self.setup_dataloaders()

            # Training loop
            training_history = []
            best_eval_loss = float("inf")

            for epoch in range(self.num_train_epochs):
                epoch_start = time.time()

                # Train epoch
                train_metrics = self.train_epoch(epoch)

                # Evaluate
                eval_metrics = self.evaluate()

                # Combine metrics
                epoch_metrics = {**train_metrics, **eval_metrics}
                epoch_metrics["epoch"] = epoch + 1
                epoch_metrics["duration"] = time.time() - epoch_start

                training_history.append(epoch_metrics)

                # Log epoch summary
                logger.info(
                    f"Epoch {epoch + 1} Summary - "
                    f"Train Loss: {epoch_metrics.get('train_loss', 0):.4f}, "
                    f"Eval Loss: {epoch_metrics.get('eval_loss', 0):.4f}, "
                    f"Duration: {epoch_metrics['duration']:.2f}s"
                )

                # Save best model
                if eval_metrics.get("eval_loss", float("inf")) < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    logger.info(f"New best evaluation loss: {best_eval_loss:.4f}")

            total_training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {total_training_time:.2f}s")

            return {
                "training_history": training_history,
                "total_time": total_training_time,
                "best_eval_loss": best_eval_loss,
                "final_train_loss": (
                    training_history[-1].get("train_loss", 0) if training_history else 0
                ),
            }

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def save_model(self, output_dir: Path):
        """Save the trained model and tokenizer."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"Model saved to: {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        if not self.model:
            return {"status": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "model_size_mb": sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            / (1024 * 1024),
        }
    
    def cleanup(self):
        """Clean up resources and free memory."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.optimizer:
                del self.optimizer
            if self.scheduler:
                del self.scheduler
            if self.scaler:
                del self.scaler
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Training engine cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
