"""
SEC (Siamese network with triplet loss) Emotion Mapper
Learns an embedding space where similar emotions cluster together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SECEmotionMapper(nn.Module):
    """
    Siamese Network for learning emotion embeddings using triplet loss.
    
    Architecture:
        Text → BERT/RoBERTa Encoder → Projection Head → L2-normalized embedding
    """
    
    def __init__(
        self,
        base_encoder: str = "roberta-base",
        embedding_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize SEC emotion mapper.
        
        Args:
            base_encoder: HuggingFace model name (e.g., "roberta-base", "bert-base-uncased")
            embedding_dim: Dimension of output embeddings
            dropout: Dropout probability
            freeze_encoder: Whether to freeze the base encoder
        """
        super().__init__()
        
        self.base_encoder_name = base_encoder
        self.embedding_dim = embedding_dim
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(base_encoder)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Base encoder frozen")
        
        # Get encoder hidden size
        encoder_hidden_size = self.encoder.config.hidden_size
        
        # Projection head: encoder_dim → embedding_dim
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, embedding_dim)
        )
        
        logger.info(f"SEC Emotion Mapper initialized:")
        logger.info(f"  Base encoder: {base_encoder}")
        logger.info(f"  Encoder hidden size: {encoder_hidden_size}")
        logger.info(f"  Embedding dimension: {embedding_dim}")
        logger.info(f"  Encoder frozen: {freeze_encoder}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: encode text and project to embedding space.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            L2-normalized embeddings [batch_size, embedding_dim]
        """
        # Encode with base model
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Project to embedding space
        embeddings = self.projection_head(cls_output)  # [batch_size, embedding_dim]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_text(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Encode a batch of texts to embeddings.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer for the base encoder
            device: Device to run on
            max_length: Maximum sequence length
            
        Returns:
            Embeddings tensor [len(texts), embedding_dim]
        """
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)
        
        return embeddings


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = "euclidean"):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: "euclidean" or "cosine"
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
        logger.info(f"Triplet Loss initialized:")
        logger.info(f"  Margin: {margin}")
        logger.info(f"  Distance metric: {distance_metric}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            loss: Scalar loss value
            stats: Dictionary with loss statistics
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        loss = losses.mean()
        
        # Statistics
        stats = {
            'loss': loss.item(),
            'pos_dist_mean': pos_dist.mean().item(),
            'neg_dist_mean': neg_dist.mean().item(),
            'margin_violations': (losses > 0).float().mean().item(),  # % of triplets with violations
        }
        
        return loss, stats


def create_sec_model(config: Dict) -> Tuple[SECEmotionMapper, AutoTokenizer]:
    """
    Factory function to create SEC model and tokenizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: SEC emotion mapper
        tokenizer: Corresponding tokenizer
    """
    model = SECEmotionMapper(
        base_encoder=config.get('base_encoder', 'roberta-base'),
        embedding_dim=config.get('embedding_dim', 256),
        dropout=config.get('dropout', 0.1),
        freeze_encoder=config.get('freeze_encoder', False)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('base_encoder', 'roberta-base')
    )
    
    return model, tokenizer


if __name__ == "__main__":
    # Test the model
    import torch
    
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = SECEmotionMapper(
        base_encoder="roberta-base",
        embedding_dim=256
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    embeddings = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embeddings are normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(batch_size))}")
    
    # Test triplet loss
    criterion = TripletLoss(margin=1.0)
    
    anchor = torch.randn(batch_size, 256)
    positive = anchor + 0.1 * torch.randn(batch_size, 256)  # Similar to anchor
    negative = torch.randn(batch_size, 256)  # Different from anchor
    
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    
    loss, stats = criterion(anchor, positive, negative)
    
    print(f"\nTriplet loss: {loss.item():.4f}")
    print(f"Statistics: {stats}")