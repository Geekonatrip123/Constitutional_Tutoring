"""
Utility script to inspect enriched DailyDialog data.
"""

import json
from pathlib import Path

def inspect_enriched_data(jsonl_path: str, num_examples: int = 3):
    """
    Display sample enriched dialogues.
    
    Args:
        jsonl_path: Path to enriched JSONL file
        num_examples: Number of example dialogues to show
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {jsonl_path}")
    print(f"{'='*80}\n")
    
    emotion_map = {
        0: "neutral",
        1: "anger",
        2: "disgust",
        3: "fear",
        4: "happiness",
        5: "sadness",
        6: "surprise"
    }
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            
            dialogue = json.loads(line)
            
            print(f"Dialogue ID: {dialogue['dialogue_id']}")
            print(f"Number of turns: {len(dialogue['utterances'])}")
            print("-" * 80)
            
            for utterance in dialogue['utterances'][:5]:  # Show first 5 turns
                emotion_label = emotion_map.get(utterance['emotion'], 'unknown')
                
                print(f"\nTurn {utterance['turn']}:")
                print(f"  Text: {utterance['text']}")
                print(f"  Emotion: {emotion_label} ({utterance['emotion']})")
                print(f"  Scene: {utterance['scene']}")
                print(f"  Knowledge: {', '.join(utterance['knowledge'])}")
            
            if len(dialogue['utterances']) > 5:
                print(f"\n  ... ({len(dialogue['utterances']) - 5} more turns)")
            
            print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Inspect enriched data
    output_dir = Path("./data/processed")
    
    # Check train set
    train_path = output_dir / "train_dailydialog_enriched.jsonl"
    if train_path.exists():
        inspect_enriched_data(str(train_path), num_examples=2)
    
    # Check validation set
    val_path = output_dir / "validation_dailydialog_enriched.jsonl"
    if val_path.exists():
        inspect_enriched_data(str(val_path), num_examples=2)