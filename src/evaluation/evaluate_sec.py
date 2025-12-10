"""
Evaluation script for SEC Emotion Mapper.
Computes embedding quality metrics and visualizations.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sec_emotion_mapper import SECEmotionMapper
from src.data.triplet_dataset import TripletDataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Emotion mapping
EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}


class SECEvaluator:
    """
    Evaluator for SEC emotion mapper.
    """
    
    def __init__(
        self,
        model: SECEmotionMapper,
        tokenizer: AutoTokenizer,
        device: torch.device
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained SEC model
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        logger.info("SEC Evaluator initialized")
    
    def encode_utterances(
        self,
        enriched_data_path: str,
        max_samples: int = 5000
    ):
        """
        Encode utterances to embeddings.
        
        Args:
            enriched_data_path: Path to enriched DailyDialog JSONL
            max_samples: Maximum number of samples to encode
            
        Returns:
            embeddings: numpy array [num_samples, embedding_dim]
            labels: numpy array [num_samples] (emotion IDs)
            texts: list of text strings
        """
        utterances = []
        labels = []
        texts = []
        
        # Load data
        with open(enriched_data_path, 'r') as f:
            for line in f:
                dialogue = json.loads(line)
                for utt in dialogue['utterances']:
                    # Create enriched text
                    enriched_text = f"{utt['text']} Scene: {utt['scene']} Knowledge: {', '.join(utt['knowledge'])}"
                    
                    utterances.append(enriched_text)
                    labels.append(utt['emotion'])
                    texts.append(utt['text'])
                    
                    if len(utterances) >= max_samples:
                        break
                
                if len(utterances) >= max_samples:
                    break
        
        logger.info(f"Encoding {len(utterances)} utterances...")
        
        # Encode in batches
        embeddings_list = []
        batch_size = 64
        
        with torch.no_grad():
            for i in tqdm(range(0, len(utterances), batch_size), desc="Encoding"):
                batch_texts = utterances[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Encode
                embeddings = self.model(input_ids, attention_mask)
                embeddings_list.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list)
        labels = np.array(labels)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings, labels, texts
    
    def compute_silhouette_score(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute silhouette score for embedding quality.
        
        Args:
            embeddings: Embedding vectors [num_samples, embedding_dim]
            labels: Emotion labels [num_samples]
            
        Returns:
            Silhouette score (higher is better, range [-1, 1])
        """
        score = silhouette_score(embeddings, labels)
        logger.info(f"Silhouette score: {score:.4f}")
        return score
    
    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ):
        """
        Visualize embeddings using t-SNE.
        
        Args:
            embeddings: Embedding vectors
            labels: Emotion labels
            output_path: Path to save visualization
        """
        logger.info("Computing t-SNE...")
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot each emotion
        for emotion_id, emotion_name in EMOTION_MAP.items():
            mask = labels == emotion_id
            if mask.sum() > 0:
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    label=emotion_name,
                    alpha=0.6,
                    s=50
                )
        
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.title("SEC Emotion Embeddings (t-SNE)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved: {output_path}")
        
        plt.close()
    
    def compute_distance_statistics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """
        Compute intra-class and inter-class distance statistics.
        
        Args:
            embeddings: Embedding vectors
            labels: Emotion labels
        """
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(embeddings, embeddings, metric='euclidean')
        
        intra_distances = []
        inter_distances = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = distances[i, j]
                
                if labels[i] == labels[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        intra_mean = np.mean(intra_distances)
        intra_std = np.std(intra_distances)
        inter_mean = np.mean(inter_distances)
        inter_std = np.std(inter_distances)
        
        logger.info("Distance Statistics:")
        logger.info(f"  Intra-class (same emotion): {intra_mean:.4f} ± {intra_std:.4f}")
        logger.info(f"  Inter-class (different emotion): {inter_mean:.4f} ± {inter_std:.4f}")
        logger.info(f"  Separation ratio: {inter_mean / intra_mean:.4f}")
        
        return {
            'intra_mean': float(intra_mean),
            'intra_std': float(intra_std),
            'inter_mean': float(inter_mean),
            'inter_std': float(inter_std),
            'separation_ratio': float(inter_mean / intra_mean)
        }
    
    def evaluate(
        self,
        enriched_data_path: str,
        output_dir: str,
        max_samples: int = 5000
    ):
        """
        Run full evaluation.
        
        Args:
            enriched_data_path: Path to enriched DailyDialog
            output_dir: Directory to save results
            max_samples: Maximum samples to evaluate
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Encode utterances
        embeddings, labels, texts = self.encode_utterances(
            enriched_data_path,
            max_samples=max_samples
        )
        
        # Compute metrics
        silhouette = self.compute_silhouette_score(embeddings, labels)
        distance_stats = self.compute_distance_statistics(embeddings, labels)
        
        # Visualize
        self.visualize_embeddings(
            embeddings,
            labels,
            output_path=str(output_dir / "embeddings_tsne.png")
        )
        
        # Save results
        results = {
            'silhouette_score': float(silhouette),
            'distance_statistics': distance_stats,
            'num_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1]
        }
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved: {results_path}")
        
        return results


def main():
    """Main execution function."""
    
    # Paths
    CHECKPOINT_PATH = "./models/checkpoints/sec_mapper/best_model.pt"
    TOKENIZER_DIR = "./models/checkpoints/sec_mapper/tokenizer"
    TEST_DATA = "./src/data/processed/test_dailydialog_enriched.jsonl"
    OUTPUT_DIR = "./results/sec_evaluation"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    model_config = checkpoint['config']
    model = SECEmotionMapper(
        base_encoder=model_config['base_encoder'],
        embedding_dim=model_config['embedding_dim'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # Create evaluator
    evaluator = SECEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Evaluate
    logger.info("=" * 80)
    logger.info("EVALUATING SEC EMOTION MAPPER")
    logger.info("=" * 80)
    
    results = evaluator.evaluate(
        enriched_data_path=TEST_DATA,
        output_dir=OUTPUT_DIR,
        max_samples=5000
    )
    
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()