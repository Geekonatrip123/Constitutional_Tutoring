"""
Build training embeddings and labels for k-NN classifier.
This creates the "Emotion Map" that k-NN will search.
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sec_emotion_mapper import SECEmotionMapper
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_training_embeddings(
    sec_checkpoint_path: str,
    tokenizer_dir: str,
    enriched_data_path: str,
    output_embeddings_path: str,
    output_labels_path: str,
    device: torch.device,
    max_samples: int = None
):
    """
    Build training embeddings and labels for k-NN classifier.
    
    Args:
        sec_checkpoint_path: Path to trained SEC model
        tokenizer_dir: Path to tokenizer
        enriched_data_path: Path to enriched DailyDialog JSONL
        output_embeddings_path: Where to save embeddings (.npy)
        output_labels_path: Where to save labels (.npy)
        device: Device to run on
        max_samples: Maximum samples to process (None = all)
    """
    logger.info("=" * 80)
    logger.info("BUILDING K-NN TRAINING SET")
    logger.info("=" * 80)
    
    # Load SEC model
    logger.info("Loading SEC model...")
    checkpoint = torch.load(sec_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    model = SECEmotionMapper(
        base_encoder=model_config['base_encoder'],
        embedding_dim=model_config['embedding_dim'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # Load enriched data
    logger.info(f"Loading enriched data from: {enriched_data_path}")
    
    utterances = []
    labels = []
    
    with open(enriched_data_path, 'r') as f:
        for line in f:
            dialogue = json.loads(line)
            for utt in dialogue['utterances']:
                # Create enriched text
                enriched_text = f"{utt['text']} Scene: {utt['scene']} Knowledge: {', '.join(utt['knowledge'])}"
                
                utterances.append(enriched_text)
                labels.append(utt['emotion'])  # Store as numeric label
                
                if max_samples and len(utterances) >= max_samples:
                    break
            
            if max_samples and len(utterances) >= max_samples:
                break
    
    logger.info(f"Loaded {len(utterances)} utterances")
    
    # Encode to embeddings
    logger.info("Encoding to embeddings...")
    
    embeddings_list = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(utterances), batch_size), desc="Encoding"):
            batch_texts = utterances[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Encode
            embeddings = model(input_ids, attention_mask)
            embeddings_list.append(embeddings.cpu().numpy())
    
    # Stack embeddings
    embeddings = np.vstack(embeddings_list)
    labels = np.array(labels)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Save
    output_embeddings_path = Path(output_embeddings_path)
    output_labels_path = Path(output_labels_path)
    
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_embeddings_path, embeddings)
    np.save(output_labels_path, labels)
    
    logger.info(f"Embeddings saved: {output_embeddings_path}")
    logger.info(f"Labels saved: {output_labels_path}")
    
    logger.info("=" * 80)
    logger.info("K-NN TRAINING SET BUILT SUCCESSFULLY!")
    logger.info("=" * 80)


def main():
    """Main execution function."""
    
    # Paths
    SEC_CHECKPOINT = "./models/checkpoints/sec_mapper/best_model.pt"
    TOKENIZER_DIR = "./models/checkpoints/sec_mapper/tokenizer"
    TRAIN_ENRICHED = "./src/data/processed/train_dailydialog_enriched.jsonl"
    
    OUTPUT_EMBEDDINGS = "./src/data/processed/train_embeddings.npy" 
    OUTPUT_LABELS = "./src/data/processed/train_labels.npy"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build training set
    build_training_embeddings(
        sec_checkpoint_path=SEC_CHECKPOINT,
        tokenizer_dir=TOKENIZER_DIR,
        enriched_data_path=TRAIN_ENRICHED,
        output_embeddings_path=OUTPUT_EMBEDDINGS,
        output_labels_path=OUTPUT_LABELS,
        device=device,
        max_samples=None  # Use all training data
    )


if __name__ == "__main__":
    main()