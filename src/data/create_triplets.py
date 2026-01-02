"""
Create triplet dataset for SEC training.
Generates (anchor, positive, negative) triplets from enriched DailyDialog.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Emotion mapping: DailyDialog emotion ID → label
EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}


class TripletGenerator:
    """
    Generates triplets for SEC training from enriched DailyDialog.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize triplet generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        
        logger.info(f"Triplet Generator initialized with seed: {random_seed}")
    
    def load_enriched_data(self, jsonl_path: str) -> List[Dict]:
        """
        Load enriched DailyDialog data.
        
        Args:
            jsonl_path: Path to enriched JSONL file
            
        Returns:
            List of utterance dictionaries
        """
        utterances = []
        
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Loading enriched data"):
                dialogue = json.loads(line)
                for utt in dialogue['utterances']:
                    # Create enriched text: original + scene + knowledge
                    enriched_text = self._create_enriched_text(utt)
                    
                    utterances.append({
                        'text': utt['text'],
                        'enriched_text': enriched_text,
                        'emotion': utt['emotion'],
                        'emotion_label': EMOTION_MAP[utt['emotion']],
                        'scene': utt['scene'],
                        'knowledge': utt['knowledge']
                    })
        
        logger.info(f"Loaded {len(utterances)} utterances")
        return utterances
    
    def _create_enriched_text(self, utterance: Dict) -> str:
        """
        Create enriched text by combining original text, scene, and knowledge.
        
        Args:
            utterance: Utterance dictionary
            
        Returns:
            Enriched text string
        """
        text = utterance['text']
        scene = utterance['scene']
        knowledge = ', '.join(utterance['knowledge'])
        
        # Format: [TEXT] Scene: [SCENE] Knowledge: [KEYWORDS]
        enriched = f"{text} Scene: {scene} Knowledge: {knowledge}"
        
        return enriched
    
    def create_triplets(
        self,
        utterances: List[Dict],
        num_triplets: int = 50000,
        hard_negative_mining: bool = True
    ) -> List[Tuple[str, str, str, str, str, str]]:
        """
        Create triplets from utterances.
        
        Args:
            utterances: List of utterance dictionaries
            num_triplets: Number of triplets to generate
            hard_negative_mining: Whether to use hard negative mining
            
        Returns:
            List of (anchor_text, anchor_label, positive_text, positive_label, 
                    negative_text, negative_label) tuples
        """
        # Group utterances by emotion
        emotion_groups = defaultdict(list)
        for utt in utterances:
            emotion_groups[utt['emotion_label']].append(utt)
        
        logger.info("Emotion distribution:")
        for emotion, utts in emotion_groups.items():
            logger.info(f"  {emotion}: {len(utts)} utterances")
        
        # Generate triplets
        triplets = []
        
        for _ in tqdm(range(num_triplets), desc="Generating triplets"):
            # Select anchor emotion
            anchor_emotion = random.choice(list(emotion_groups.keys()))
            
            # Select anchor and positive (same emotion)
            if len(emotion_groups[anchor_emotion]) < 2:
                continue  # Skip if not enough samples
            
            anchor, positive = random.sample(emotion_groups[anchor_emotion], 2)
            
            # Select negative (different emotion)
            negative_emotions = [e for e in emotion_groups.keys() if e != anchor_emotion]
            if not negative_emotions:
                continue
            
            negative_emotion = random.choice(negative_emotions)
            negative = random.choice(emotion_groups[negative_emotion])
            
            # Create triplet
            triplet = (
                anchor['enriched_text'],
                anchor['emotion_label'],
                positive['enriched_text'],
                positive['emotion_label'],
                negative['enriched_text'],
                negative['emotion_label']
            )
            
            triplets.append(triplet)
        
        logger.info(f"Generated {len(triplets)} triplets")
        return triplets
    
    def save_triplets(
        self,
        triplets: List[Tuple],
        output_path: str
    ):
        """
        Save triplets to JSONL file.
        
        Args:
            triplets: List of triplets
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for triplet in triplets:
                anchor_text, anchor_label, pos_text, pos_label, neg_text, neg_label = triplet
                
                data = {
                    'anchor': {
                        'text': anchor_text,
                        'emotion': anchor_label
                    },
                    'positive': {
                        'text': pos_text,
                        'emotion': pos_label
                    },
                    'negative': {
                        'text': neg_text,
                        'emotion': neg_label
                    }
                }
                
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Saved triplets to: {output_path}")


def main():
    """Main execution function."""
    
    # Paths - UPDATED
    PROCESSED_DIR = Path(r"C:\Users\shlok\Research\src\data\processed")
    TRAIN_ENRICHED = PROCESSED_DIR / "train_dailydialog_enriched_balanced.jsonl"  # ← CHANGED
    VAL_ENRICHED = PROCESSED_DIR / "validation_dailydialog_enriched.jsonl"
    
    TRAIN_TRIPLETS = PROCESSED_DIR / "train_triplets_balanced.jsonl"  # ← CHANGED output name
    VAL_TRIPLETS = PROCESSED_DIR / "val_triplets.jsonl"
    
    # Initialize generator
    generator = TripletGenerator(random_seed=42)
    
    # Generate training triplets
    logger.info("=" * 80)
    logger.info("GENERATING TRAINING TRIPLETS (FROM BALANCED DATA)")
    logger.info("=" * 80)
    
    train_utterances = generator.load_enriched_data(str(TRAIN_ENRICHED))
    train_triplets = generator.create_triplets(
        utterances=train_utterances,
        num_triplets=50000,
        hard_negative_mining=True
    )
    generator.save_triplets(train_triplets, str(TRAIN_TRIPLETS))
    
    # Generate validation triplets
    logger.info("=" * 80)
    logger.info("GENERATING VALIDATION TRIPLETS")
    logger.info("=" * 80)
    
    val_utterances = generator.load_enriched_data(str(VAL_ENRICHED))
    val_triplets = generator.create_triplets(
        utterances=val_utterances,
        num_triplets=5000,
        hard_negative_mining=True
    )
    generator.save_triplets(val_triplets, str(VAL_TRIPLETS))
    
    logger.info("=" * 80)
    logger.info("TRIPLET GENERATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()