# test_classifier_on_test_set.py
"""
Test the complete classifier on the enriched test set.
"""

import torch
import numpy as np
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.models.sec_emotion_mapper import SECEmotionMapper
from transformers import AutoTokenizer

# Emotion mappings
EMOTION_MAP = {
    0: "neutral", 1: "anger", 2: "disgust", 
    3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"
}

EMOTION_TO_PEDAGOGICAL = {
    "sadness": "confused",
    "fear": "confused",
    "anger": "frustrated",
    "happiness": "correct",
    "disgust": "disengaged",
    "neutral": "neutral",
    "surprise": "neutral"
}

print("="*80)
print("TESTING STUDENT STATE CLASSIFIER ON TEST SET")
print("="*80)

# Load SEC model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nLoading SEC model on {device}...")

checkpoint = torch.load("./models/checkpoints/sec_mapper/best_model.pt", map_location=device)
model = SECEmotionMapper(
    base_encoder="roberta-base",
    embedding_dim=256,
    dropout=0.1
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./models/checkpoints/sec_mapper/tokenizer")

# Load training embeddings and labels for k-NN
print("Loading training embeddings...")
train_embeddings = np.load("./src/data/processed/train_embeddings.npy")
train_labels = np.load("./src/data/processed/train_labels.npy")
train_labels_str = [EMOTION_MAP[int(label)] for label in train_labels]

print(f"Training set: {len(train_embeddings)} samples")

# Load test set
print("\nLoading test set...")
test_utterances = []

with open("./src/data/processed/test_dailydialog_enriched.jsonl", 'r') as f:
    for line in f:
        dialogue = json.loads(line)
        for utt in dialogue['utterances']:
            # Create enriched text
            enriched_text = f"{utt['text']} Scene: {utt['scene']} Knowledge: {', '.join(utt['knowledge'])}"
            
            test_utterances.append({
                'text': utt['text'],
                'enriched_text': enriched_text,
                'true_emotion': utt['emotion'],
                'true_emotion_label': EMOTION_MAP[utt['emotion']],
                'true_pedagogical_state': EMOTION_TO_PEDAGOGICAL[EMOTION_MAP[utt['emotion']]]
            })

print(f"Test set: {len(test_utterances)} utterances")

# Encode test utterances
print("\nEncoding test utterances...")
test_embeddings = []
batch_size = 64

with torch.no_grad():
    for i in tqdm(range(0, len(test_utterances), batch_size)):
        batch = test_utterances[i:i+batch_size]
        batch_texts = [utt['enriched_text'] for utt in batch]
        
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        embeddings = model(input_ids, attention_mask)
        test_embeddings.append(embeddings.cpu().numpy())

test_embeddings = np.vstack(test_embeddings)
print(f"Test embeddings shape: {test_embeddings.shape}")

# k-NN Classification
print("\nRunning k-NN classification (k=5)...")
k = 5
predictions = []

for test_emb in tqdm(test_embeddings):
    # Compute distances
    distances = cdist(test_emb.reshape(1, -1), train_embeddings, metric='euclidean').flatten()
    
    # Find k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = [train_labels_str[i] for i in k_nearest_indices]
    
    # Vote
    label_counts = Counter(k_nearest_labels)
    predicted_emotion = label_counts.most_common(1)[0][0]
    predicted_pedagogical = EMOTION_TO_PEDAGOGICAL[predicted_emotion]
    
    predictions.append({
        'predicted_emotion': predicted_emotion,
        'predicted_pedagogical': predicted_pedagogical
    })

# Evaluate
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

# DailyDialog Emotion Accuracy
emotion_correct = sum(
    1 for i, pred in enumerate(predictions) 
    if pred['predicted_emotion'] == test_utterances[i]['true_emotion_label']
)
emotion_accuracy = emotion_correct / len(predictions)

print(f"\nDailyDialog Emotion Accuracy: {emotion_accuracy:.2%} ({emotion_correct}/{len(predictions)})")

# Pedagogical State Accuracy
pedagogical_correct = sum(
    1 for i, pred in enumerate(predictions)
    if pred['predicted_pedagogical'] == test_utterances[i]['true_pedagogical_state']
)
pedagogical_accuracy = pedagogical_correct / len(predictions)

print(f"Pedagogical State Accuracy: {pedagogical_accuracy:.2%} ({pedagogical_correct}/{len(predictions)})")

# Confusion matrix for pedagogical states
print("\nPedagogical State Predictions:")
pedagogical_states = list(set(EMOTION_TO_PEDAGOGICAL.values()))

for state in pedagogical_states:
    true_count = sum(1 for utt in test_utterances if utt['true_pedagogical_state'] == state)
    pred_count = sum(1 for pred in predictions if pred['predicted_pedagogical'] == state)
    correct_count = sum(
        1 for i, pred in enumerate(predictions)
        if test_utterances[i]['true_pedagogical_state'] == state 
        and pred['predicted_pedagogical'] == state
    )
    
    if true_count > 0:
        recall = correct_count / true_count
        print(f"  {state}: {true_count} samples, {correct_count} correct, recall={recall:.2%}")

# Show some examples
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

for i in range(min(10, len(test_utterances))):
    utt = test_utterances[i]
    pred = predictions[i]
    
    print(f"\nUtterance: {utt['text']}")
    print(f"True: {utt['true_pedagogical_state'].upper()} (emotion: {utt['true_emotion_label']})")
    print(f"Predicted: {pred['predicted_pedagogical'].upper()} (emotion: {pred['predicted_emotion']})")
    
    if pred['predicted_pedagogical'] == utt['true_pedagogical_state']:
        print("✓ CORRECT")
    else:
        print("✗ WRONG")
    print("-" * 80)

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)