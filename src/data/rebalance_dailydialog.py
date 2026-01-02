"""
Rebalance DailyDialog training data to reduce neutral bias.
"""

import pandas as pd
from collections import Counter
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebalance_dailydialog(
    input_csv: str = r"C:\Users\shlok\Research\src\data\raw\dailydialog\train_daily.csv",
    output_csv: str = r"C:\Users\shlok\Research\src\data\raw\dailydialog\train_daily_balanced.csv",
    neutral_ratio: float = 0.5
):
    """
    Rebalance DailyDialog to get ~50% neutral, ~50% other emotions.
    
    Args:
        input_csv: Original train_daily.csv
        output_csv: Output balanced CSV
        neutral_ratio: Target neutral ratio (0.5 = 50%)
    """
    logger.info("=" * 80)
    logger.info("REBALANCING DAILYDIALOG TRAINING DATA")
    logger.info("=" * 80)
    
    logger.info(f"\nLoading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    logger.info(f"Original: {len(df)} dialogues")
    
    # Count emotions across all dialogues
    all_emotions = []
    for dialog_str in df['emotion']:
        # Parse space-separated emotion string
        # Format: "[0 0 0 0 0 0 4 4 4 4]" → [0, 0, 0, 0, 0, 0, 4, 4, 4, 4]
        emotion_str = str(dialog_str).strip('[]')
        emotions = [int(x) for x in emotion_str.split()]
        all_emotions.extend(emotions)
    
    original_counts = Counter(all_emotions)
    logger.info("\nOriginal emotion distribution:")
    for emotion_id, count in sorted(original_counts.items()):
        percentage = (count / len(all_emotions)) * 100
        logger.info(f"  Emotion {emotion_id}: {count:6d} ({percentage:5.1f}%)")
    
    # Strategy: Keep dialogues based on their emotion composition
    # We'll undersample dialogues that are mostly neutral
    
    neutral_dialogues = []
    non_neutral_dialogues = []
    
    for idx, row in df.iterrows():
        # Parse emotions
        emotion_str = str(row['emotion']).strip('[]')
        emotions = [int(x) for x in emotion_str.split()]
        
        neutral_count = sum(1 for e in emotions if e == 0)
        neutral_pct = neutral_count / len(emotions) if len(emotions) > 0 else 0
        
        if neutral_pct > 0.7:  # Mostly neutral
            neutral_dialogues.append(idx)
        else:
            non_neutral_dialogues.append(idx)
    
    logger.info(f"\nDialogue classification:")
    logger.info(f"  Mostly neutral dialogues: {len(neutral_dialogues)}")
    logger.info(f"  Non-neutral dialogues: {len(non_neutral_dialogues)}")
    
    # Calculate how many neutral dialogues to keep
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
    
    # Combine indices
    selected_indices = sampled_neutral + non_neutral_dialogues
    random.shuffle(selected_indices)
    
    # Create balanced dataframe
    balanced_df = df.iloc[selected_indices].reset_index(drop=True)
    
    # Count final distribution
    final_emotions = []
    for dialog_str in balanced_df['emotion']:
        emotion_str = str(dialog_str).strip('[]')
        emotions = [int(x) for x in emotion_str.split()]
        final_emotions.extend(emotions)
    
    final_counts = Counter(final_emotions)
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EMOTION DISTRIBUTION:")
    logger.info("=" * 60)
    for emotion_id, count in sorted(final_counts.items()):
        percentage = (count / len(final_emotions)) * 100
        logger.info(f"  Emotion {emotion_id}: {count:6d} ({percentage:5.1f}%)")
    
    logger.info(f"\nFinal: {len(balanced_df)} dialogues ({len(final_emotions)} total utterances)")
    
    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_csv, index=False)
    logger.info(f"\n✅ Saved to: {output_csv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Update enrich_dailydialog_cosike.py to use:")
    logger.info(r"   C:\Users\shlok\Research\src\data\raw\dailydialog\train_daily_balanced.csv")
    logger.info("")
    logger.info("2. Run: python src\\data\\enrich_dailydialog_cosike.py")
    logger.info("3. Run: python src\\data\\create_triplets.py")
    logger.info("4. Retrain SEC model")
    logger.info("5. Run: python src\\data\\build_knn_training_set.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    rebalance_dailydialog()