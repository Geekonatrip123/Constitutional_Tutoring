"""
Training script for SEC Emotion Mapper.
Trains Siamese Network with triplet loss on enriched DailyDialog.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sec_emotion_mapper import SECEmotionMapper, TripletLoss, create_sec_model
from src.data.triplet_dataset import create_triplet_dataloader
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SECTrainer:
    """
    Trainer for SEC Emotion Mapper.
    """
    
    def __init__(
        self,
        model: SECEmotionMapper,
        tokenizer: AutoTokenizer,
        criterion: TripletLoss,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: dict,
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: SEC emotion mapper model
            tokenizer: Tokenizer
            criterion: Triplet loss function
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * config['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_steps = config.get('save_steps', 1000)
        self.eval_steps = config.get('eval_steps', 500)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        logger.info("Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Total training steps: {total_steps}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            anchor_ids = batch['anchor_input_ids'].to(self.device)
            anchor_mask = batch['anchor_attention_mask'].to(self.device)
            positive_ids = batch['positive_input_ids'].to(self.device)
            positive_mask = batch['positive_attention_mask'].to(self.device)
            negative_ids = batch['negative_input_ids'].to(self.device)
            negative_mask = batch['negative_attention_mask'].to(self.device)
            
            # Forward pass
            anchor_emb = self.model(anchor_ids, anchor_mask)
            positive_emb = self.model(positive_ids, positive_mask)
            negative_emb = self.model(negative_ids, negative_mask)
            
            # Compute loss
            loss, stats = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_dist': f"{stats['pos_dist_mean']:.4f}",
                'neg_dist': f"{stats['neg_dist_mean']:.4f}",
                'violations': f"{stats['margin_violations']:.2%}"
            })
            
            # Evaluation
            if self.global_step % self.eval_steps == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.global_step, val_loss))
                
                logger.info(
                    f"Step {self.global_step} - "
                    f"Train Loss: {loss.item():.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Save if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                
                self.model.train()
            
            # Regular checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(is_best=False)
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_losses.append((epoch, avg_loss))
        
        logger.info(f"Epoch {epoch + 1} - Average Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self):
        """
        Evaluate on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                anchor_ids = batch['anchor_input_ids'].to(self.device)
                anchor_mask = batch['anchor_attention_mask'].to(self.device)
                positive_ids = batch['positive_input_ids'].to(self.device)
                positive_mask = batch['positive_attention_mask'].to(self.device)
                negative_ids = batch['negative_input_ids'].to(self.device)
                negative_mask = batch['negative_attention_mask'].to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor_ids, anchor_mask)
                positive_emb = self.model(positive_ids, positive_mask)
                negative_emb = self.model(negative_ids, negative_mask)
                
                # Compute loss
                loss, _ = self.criterion(anchor_emb, positive_emb, negative_emb)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save tokenizer
        tokenizer_dir = self.checkpoint_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
    
    def save_training_stats(self):
        """Save training statistics."""
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step
        }
        
        stats_path = self.checkpoint_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training stats saved: {stats_path}")
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        for epoch in range(self.config['num_epochs']):
            avg_loss = self.train_epoch(epoch)
            
            # Evaluate at end of epoch
            val_loss = self.evaluate()
            self.val_losses.append((self.global_step, val_loss))
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Save if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                logger.info(f"New best validation loss: {val_loss:.4f}")
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        self.save_training_stats()
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)


def main():
    """Main execution function."""
    
    # Load configuration
    config = load_config()
    train_config = config.sec_training.to_dict()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model and tokenizer
    logger.info("Creating model...")
    model, tokenizer = create_sec_model(train_config)
    
    # Create criterion
    criterion = TripletLoss(
        margin=train_config['margin'],
        distance_metric=train_config['distance_metric']
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_triplet_dataloader(
        triplets_jsonl="./src/data/processed/train_triplets.jsonl",
        tokenizer=tokenizer,
        batch_size=train_config['batch_size'],
        max_length=128,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = create_triplet_dataloader(
        triplets_jsonl="./src/data/processed/val_triplets.jsonl",
        tokenizer=tokenizer,
        batch_size=train_config['batch_size'],
        max_length=128,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Val batches: {len(val_dataloader)}")
    
    # Create trainer
    trainer = SECTrainer(
        model=model,
        tokenizer=tokenizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()