"""
PyTorch Dataset for triplet training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """
    Dataset for triplet training with SEC.
    Returns tokenized (anchor, positive, negative) triplets.
    """
    
    def __init__(
        self,
        triplets_jsonl: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        """
        Initialize triplet dataset.
        
        Args:
            triplets_jsonl: Path to triplets JSONL file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load triplets
        self.triplets = []
        with open(triplets_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.triplets.append(data)
        
        logger.info(f"Loaded {len(self.triplets)} triplets from {triplets_jsonl}")
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet.
        
        Returns:
            Dictionary with tokenized anchor, positive, and negative texts
        """
        triplet = self.triplets[idx]
        
        # Extract texts
        anchor_text = triplet['anchor']['text']
        positive_text = triplet['positive']['text']
        negative_text = triplet['negative']['text']
        
        # Tokenize
        anchor_encoded = self.tokenizer(
            anchor_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoded = self.tokenizer(
            positive_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_encoded = self.tokenizer(
            negative_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoded['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoded['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoded['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoded['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoded['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoded['attention_mask'].squeeze(0),
        }


def create_triplet_dataloader(
    triplets_jsonl: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for triplet training.
    
    Args:
        triplets_jsonl: Path to triplets JSONL file
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    dataset = TripletDataset(
        triplets_jsonl=triplets_jsonl,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Create dataset
    dataset = TripletDataset(
        triplets_jsonl="./data/processed/train_triplets.jsonl",
        tokenizer=tokenizer,
        max_length=128
    )
    
    # Test __getitem__
    sample = dataset[0]
    
    print("Sample triplet:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")
    
    # Test dataloader
    dataloader = create_triplet_dataloader(
        triplets_jsonl="./data/processed/train_triplets.jsonl",
        tokenizer=tokenizer,
        batch_size=4,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    print("\nBatch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")