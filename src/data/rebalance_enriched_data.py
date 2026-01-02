"""
Rebalance ENRICHED DailyDialog data (after COSIKE enrichment).
"""

import json
from collections import Counter
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebalance_enriched_dialogues(
    input_jsonl: str = r"C:\Users\shlok\Research\src\data\processed\train_dailydialog_enriched.jsonl",
    output_jsonl: str = r"C:\Users\shlok\Research\src\data\processed\train_dailydialog_enriched_balanced.jsonl",
    neutral_ratio: float = 0.5
):
    """
    Rebalance enriched dialogues to reduce neutral bias.
    
    Args:
        input_jsonl: Input enriched JSONL file
        output_jsonl: Output balanced enriched JSONL file
        neutral_ratio: Target neutral ratio (0.5 = 50%)
    """
    logger.info("=" * 80)
    logger.info("REBALANCING ENRICHED DIALOGUES")
    logger.info("=" * 80)
    
    logger.info(f"\nLoading {input_jsonl}...")
    
    # Load all dialogues
    dialogues = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))
    
    logger.info(f"Loaded {len(dialogues)} dialogues")
    
    # Count emotions across all dialogues
    all_emotions = []
    for dialogue in dialogues:
        for utt in dialogue['utterances']:
            all_emotions.append(utt['emotion'])
    
    original_counts = Counter(all_emotions)
    logger.info("\nOriginal emotion distribution:")
    for emotion_id, count in sorted(original_counts.items()):
        percentage = (count / len(all_emotions)) * 100
        logger.info(f"  Emotion {emotion_id}: {count:6d} ({percentage:5.1f}%)")
    
    # Classify dialogues
    neutral_dialogues = []
    non_neutral_dialogues = []
    
    for dialogue in dialogues:
        emotions = [utt['emotion'] for utt in dialogue['utterances']]
        neutral_count = sum(1 for e in emotions if e == 0)
        neutral_pct = neutral_count / len(emotions) if len(emotions) > 0 else 0
        
        if neutral_pct > 0.7:  # Mostly neutral
            neutral_dialogues.append(dialogue)
        else:
            non_neutral_dialogues.append(dialogue)
    
    logger.info(f"\nDialogue classification:")
    logger.info(f"  Mostly neutral dialogues: {len(neutral_dialogues)}")
    logger.info(f"  Non-neutral dialogues: {len(non_neutral_dialogues)}")
    
    # Calculate target
    non_neutral_count = len(non_neutral_dialogues)
    total_target = int(non_neutral_count / (1 - neutral_ratio))
    neutral_target = total_target - non_neutral_count
    
    logger.info(f"\nTarget:")
    logger.info(f"  Total dialogues: {total_target}")
    logger.info(f"  Neutral dialogues: {neutral_target}")
    logger.info(f"  Non-neutral dialogues: {non_neutral_count}")
    
    # Sample neutral dialogues
    random.seed(42)
    
    if neutral_target < len(neutral_dialogues):
        logger.info(f"\nUndersampling neutral dialogues: {len(neutral_dialogues)} → {neutral_target}")
        sampled_neutral = random.sample(neutral_dialogues, neutral_target)
    else:
        sampled_neutral = neutral_dialogues
    
    # Combine and shuffle
    balanced_dialogues = sampled_neutral + non_neutral_dialogues
    random.shuffle(balanced_dialogues)
    
    # Count final distribution
    final_emotions = []
    for dialogue in balanced_dialogues:
        for utt in dialogue['utterances']:
            final_emotions.append(utt['emotion'])
    
    final_counts = Counter(final_emotions)
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EMOTION DISTRIBUTION:")
    logger.info("=" * 60)
    for emotion_id, count in sorted(final_counts.items()):
        percentage = (count / len(final_emotions)) * 100
        logger.info(f"  Emotion {emotion_id}: {count:6d} ({percentage:5.1f}%)")
    
    logger.info(f"\nFinal: {len(balanced_dialogues)} dialogues ({len(final_emotions)} total utterances)")
    
    # Save
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w') as f:
        for dialogue in balanced_dialogues:
            f.write(json.dumps(dialogue) + '\n')
    
    logger.info(f"\n✅ Saved to: {output_jsonl}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Run: python src\\data\\create_triplets.py")
    logger.info("   (Update to use train_dailydialog_enriched_balanced.jsonl)")
    logger.info("2. Retrain SEC model")
    logger.info("3. Run: python src\\data\\build_knn_training_set.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    rebalance_enriched_dialogues()