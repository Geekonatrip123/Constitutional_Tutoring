"""
Complete Student State Classifier: COSIKE + SEC + k-NN
This is the final inference pipeline that takes a raw utterance and outputs a pedagogical state.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import logging
from collections import Counter
from scipy.spatial.distance import cdist

from transformers import AutoTokenizer
from src.models.sec_emotion_mapper import SECEmotionMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Emotion mapping: DailyDialog → Pedagogical States
DAILYDIALOG_EMOTIONS = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

# Mapping to pedagogical states (based on our config)
EMOTION_TO_PEDAGOGICAL = {
    "sadness": "confused",
    "fear": "confused",
    "anger": "frustrated",
    "happiness": "correct",
    "disgust": "disengaged",
    "neutral": "neutral",
    "surprise": "neutral"
}


class COSIKEEnricher:
    """
    Phase 1: COSIKE Enrichment
    Adds scene description and commonsense knowledge to utterances.
    """
    
    def __init__(self, llm_api, use_cache: bool = True, cache_dir: str = "./data/processed/cosike_cache"):
        """
        Initialize COSIKE enricher.
        
        Args:
            llm_api: LLM API client (Gemini or Qwen)
            use_cache: Whether to cache enrichments
            cache_dir: Directory for caching
        """
        self.llm_api = llm_api
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = self._load_cache()
        else:
            self.cache = {}
        
        logger.info("COSIKE Enricher initialized")
    
    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        cache_file = self.cache_dir / "enrichment_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached enrichments")
            return cache
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "enrichment_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def enrich(self, utterance: str) -> Dict[str, any]:
        """
        Enrich a single utterance with scene and knowledge.
        
        Args:
            utterance: Raw text utterance
            
        Returns:
            Dictionary with 'scene' and 'knowledge' fields
        """
        # Check cache
        if self.use_cache and utterance in self.cache:
            return self.cache[utterance]
        
        # Generate enrichment via LLM
        prompt = f"""Given the following utterance in a conversational context:
"{utterance}"

Generate:
1. Scene Description: A brief description of the conversational context, with specific emphasis on the emotional tone and any feelings being expressed by the speaker (1-2 sentences)
2. Commonsense Knowledge: Relevant keywords or concepts related to the utterance (5-7 keywords)

Output as JSON with keys: "scene" and "knowledge" (knowledge should be a list)
"""
        
        try:
            # Call LLM
            response = self.llm_api.generate_json(prompt, temperature=0.7, max_tokens=256)
            
            enrichment = {
                'scene': response.get('scene', 'Unable to determine scene'),
                'knowledge': response.get('knowledge', ['unknown'])
            }
            
            # Cache result
            if self.use_cache:
                self.cache[utterance] = enrichment
                if len(self.cache) % 100 == 0:  # Save every 100 new entries
                    self._save_cache()
            
            return enrichment
            
        except Exception as e:
            logger.warning(f"Enrichment failed: {e}, using defaults")
            return {
                'scene': 'Unable to determine scene',
                'knowledge': ['unknown']
            }
    
    def create_enriched_text(self, utterance: str, scene: str, knowledge: List[str]) -> str:
        """
        Combine utterance, scene, and knowledge into enriched text.
        
        Args:
            utterance: Original text
            scene: Scene description
            knowledge: List of knowledge keywords
            
        Returns:
            Enriched text string
        """
        knowledge_str = ', '.join(knowledge)
        enriched = f"{utterance} Scene: {scene} Knowledge: {knowledge_str}"
        return enriched


class SECEmbedder:
    """
    Phase 2: SEC Embedding
    Encodes enriched text into embedding space using trained SEC model.
    """
    
    def __init__(
        self,
        model: SECEmotionMapper,
        tokenizer: AutoTokenizer,
        device: torch.device
    ):
        """
        Initialize SEC embedder.
        
        Args:
            model: Trained SEC model
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        logger.info("SEC Embedder initialized")
    
    def embed(self, enriched_text: str, max_length: int = 128) -> np.ndarray:
        """
        Embed enriched text into vector space.
        
        Args:
            enriched_text: Text with scene and knowledge
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector [embedding_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            enriched_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Encode
        with torch.no_grad():
            embedding = self.model(input_ids, attention_mask)
        
        return embedding.cpu().numpy().flatten()


class kNNClassifier:
    """
    Phase 3: k-NN Classification
    Classifies based on nearest neighbors in embedding space.
    """
    
    def __init__(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        k: int = 5,
        distance_metric: str = "euclidean"
    ):
        """
        Initialize k-NN classifier.
        
        Args:
            train_embeddings: Training embeddings [num_samples, embedding_dim]
            train_labels: Training emotion labels [num_samples]
            k: Number of neighbors
            distance_metric: Distance metric ("euclidean" or "cosine")
        """
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.k = k
        self.distance_metric = distance_metric
        
        logger.info(f"k-NN Classifier initialized:")
        logger.info(f"  Training samples: {len(train_embeddings)}")
        logger.info(f"  k: {k}")
        logger.info(f"  Distance metric: {distance_metric}")
    
    def predict(self, query_embedding: np.ndarray) -> str:
        """
        Predict emotion label for query embedding.
        
        Args:
            query_embedding: Query embedding vector [embedding_dim]
            
        Returns:
            Predicted emotion label (DailyDialog emotion name)
        """
        # Compute distances to all training samples
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.distance_metric == "euclidean":
            distances = cdist(query_embedding, self.train_embeddings, metric='euclidean').flatten()
        elif self.distance_metric == "cosine":
            distances = cdist(query_embedding, self.train_embeddings, metric='cosine').flatten()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Find k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.train_labels[i] for i in k_nearest_indices]
        
        # Vote: most common label
        label_counts = Counter(k_nearest_labels)
        predicted_label = label_counts.most_common(1)[0][0]
        
        return predicted_label


class StudentStateClassifier:
    """
    Complete Student State Classifier Pipeline.
    
    Pipeline:
        Raw utterance → COSIKE enrichment → SEC embedding → k-NN classification → Pedagogical state
    """
    
    def __init__(
        self,
        cosike_enricher: COSIKEEnricher,
        sec_embedder: SECEmbedder,
        knn_classifier: kNNClassifier,
        emotion_to_pedagogical_map: Dict[str, str] = EMOTION_TO_PEDAGOGICAL
    ):
        """
        Initialize complete classifier.
        
        Args:
            cosike_enricher: COSIKE enricher
            sec_embedder: SEC embedder
            knn_classifier: k-NN classifier
            emotion_to_pedagogical_map: Mapping from DailyDialog emotions to pedagogical states
        """
        self.cosike_enricher = cosike_enricher
        self.sec_embedder = sec_embedder
        self.knn_classifier = knn_classifier
        self.emotion_map = emotion_to_pedagogical_map
        
        logger.info("Complete Student State Classifier initialized")
    
    def predict(self, utterance: str) -> str:
        """
        Predict pedagogical state for a student utterance.
        
        Args:
            utterance: Raw student utterance text
            
        Returns:
            Pedagogical state label: one of {"confused", "frustrated", "correct", "disengaged", "neutral"}
        """
        # Phase 1: COSIKE enrichment
        enrichment = self.cosike_enricher.enrich(utterance)
        enriched_text = self.cosike_enricher.create_enriched_text(
            utterance,
            enrichment['scene'],
            enrichment['knowledge']
        )
        
        # Phase 2: SEC embedding
        embedding = self.sec_embedder.embed(enriched_text)
        
        # Phase 3: k-NN classification
        dailydialog_emotion = self.knn_classifier.predict(embedding)
        
        # Map to pedagogical state
        pedagogical_state = self.emotion_map.get(dailydialog_emotion, "neutral")
        
        return pedagogical_state
    
    def predict_with_details(self, utterance: str) -> Dict[str, any]:
        """
        Predict with full details for debugging/analysis.
        
        Args:
            utterance: Raw student utterance
            
        Returns:
            Dictionary with all intermediate outputs
        """
        # Phase 1: COSIKE enrichment
        enrichment = self.cosike_enricher.enrich(utterance)
        enriched_text = self.cosike_enricher.create_enriched_text(
            utterance,
            enrichment['scene'],
            enrichment['knowledge']
        )
        
        # Phase 2: SEC embedding
        embedding = self.sec_embedder.embed(enriched_text)
        
        # Phase 3: k-NN classification
        dailydialog_emotion = self.knn_classifier.predict(embedding)
        
        # Map to pedagogical state
        pedagogical_state = self.emotion_map.get(dailydialog_emotion, "neutral")
        
        return {
            'utterance': utterance,
            'enrichment': enrichment,
            'enriched_text': enriched_text,
            'embedding_norm': float(np.linalg.norm(embedding)),
            'dailydialog_emotion': dailydialog_emotion,
            'pedagogical_state': pedagogical_state
        }


def load_student_state_classifier(
    sec_checkpoint_path: str,
    tokenizer_dir: str,
    train_embeddings_path: str,
    train_labels_path: str,
    llm_api,
    device: torch.device,
    k: int = 5
) -> StudentStateClassifier:
    """
    Load complete student state classifier from saved components.
    
    Args:
        sec_checkpoint_path: Path to SEC model checkpoint
        tokenizer_dir: Path to tokenizer directory
        train_embeddings_path: Path to training embeddings .npy file
        train_labels_path: Path to training labels .npy file
        llm_api: LLM API client for COSIKE
        device: Device to run on
        k: Number of neighbors for k-NN
        
    Returns:
        Complete StudentStateClassifier
    """
    logger.info("Loading Student State Classifier...")
    
    # Load SEC model
    logger.info("Loading SEC model...")
    checkpoint = torch.load(sec_checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    sec_model = SECEmotionMapper(
        base_encoder=model_config['base_encoder'],
        embedding_dim=model_config['embedding_dim'],
        dropout=model_config['dropout']
    )
    sec_model.load_state_dict(checkpoint['model_state_dict'])
    sec_model.to(device)
    sec_model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # Load training embeddings and labels
    logger.info("Loading training embeddings and labels...")
    train_embeddings = np.load(train_embeddings_path)
    train_labels_raw = np.load(train_labels_path)
    
    # Convert numeric labels to emotion names
    train_labels = [DAILYDIALOG_EMOTIONS[int(label)] for label in train_labels_raw]
    
    logger.info(f"Training set: {len(train_embeddings)} samples")
    
    # Create components
    cosike_enricher = COSIKEEnricher(llm_api=llm_api, use_cache=True)
    sec_embedder = SECEmbedder(model=sec_model, tokenizer=tokenizer, device=device)
    knn_classifier = kNNClassifier(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        k=k,
        distance_metric="euclidean"
    )
    
    # Create complete classifier
    classifier = StudentStateClassifier(
        cosike_enricher=cosike_enricher,
        sec_embedder=sec_embedder,
        knn_classifier=knn_classifier
    )
    
    logger.info("Student State Classifier loaded successfully!")
    
    return classifier


if __name__ == "__main__":
    # Test the complete pipeline
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.utils.llm_utils import GeminiAPI
    from src.utils.config import load_config
    
    # Load config
    config = load_config()
    
    # Create Gemini API
    gemini = GeminiAPI(config.api_keys.gemini.to_dict())
    
    # Load classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = load_student_state_classifier(
        sec_checkpoint_path="./models/checkpoints/sec_mapper/best_model.pt",
        tokenizer_dir="./models/checkpoints/sec_mapper/tokenizer",
        train_embeddings_path="./src/data/processed/train_embeddings.npy",
        train_labels_path="./src/data/processed/train_labels.npy",
        llm_api=gemini,
        device=device,
        k=5
    )
    
    # Test predictions
    test_utterances = [
        "I don't understand what you mean by derivative",
        "This is so frustrating! I can't figure this out!",
        "Oh I get it now! That makes sense!",
        "Whatever, I don't care anymore",
        "Can you explain that again?"
    ]
    
    print("\n" + "="*80)
    print("TESTING STUDENT STATE CLASSIFIER")
    print("="*80 + "\n")
    
    for utterance in test_utterances:
        result = classifier.predict_with_details(utterance)
        
        print(f"Utterance: {result['utterance']}")
        print(f"Scene: {result['enrichment']['scene']}")
        print(f"Knowledge: {', '.join(result['enrichment']['knowledge'])}")
        print(f"DailyDialog Emotion: {result['dailydialog_emotion']}")
        print(f"→ Pedagogical State: {result['pedagogical_state'].upper()}")
        print("-" * 80 + "\n")