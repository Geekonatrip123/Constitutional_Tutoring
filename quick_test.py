"""Test with REAL Gemini enrichment"""
import torch
import numpy as np
import json
from collections import Counter
from scipy.spatial.distance import cdist
from src.models.sec_emotion_mapper import SECEmotionMapper
from transformers import AutoTokenizer
import google.generativeai as genai

# Configure Gemini directly
API_KEY = "AIzaSyCOJ2GlTMWvgbZBlD0DC38EPqy_NZ_dGoU"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Emotion mappings
EMOTION_MAP = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}
EMOTION_TO_PEDAGOGICAL = {
    "sadness": "confused", "fear": "confused", "anger": "frustrated",
    "happiness": "correct", "disgust": "disengaged", "neutral": "neutral", "surprise": "neutral"
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("./models/checkpoints/sec_mapper/best_model.pt", map_location=device)
model = SECEmotionMapper(base_encoder="roberta-base", embedding_dim=256, dropout=0.1)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./models/checkpoints/sec_mapper/tokenizer")

# Load training data
train_embeddings = np.load("./src/data/processed/train_embeddings.npy")
train_labels = np.load("./src/data/processed/train_labels.npy")
train_labels_str = [EMOTION_MAP[int(label)] for label in train_labels]

# Test utterances (student-like)
test_utterances = [
    "I am so angry!",
    "I am so sad",
    "I am so happy!",
    "I don't understand this at all! It makes no sense!",
    "Oh I see! Now I get it!",
    "This is impossible. I give up.",
    "I'm confused about derivatives",
    "This is so frustrating!",
    "I finally got it right!",
]

print("="*80)
print("TESTING WITH REAL GEMINI ENRICHMENT")
print("="*80)

for utterance in test_utterances:
    print(f"\nOriginal: \"{utterance}\"")
    
    # Get REAL Gemini enrichment
    prompt = f"""Given the following utterance: "{utterance}"

Generate:
1. Scene Description: Brief description with emphasis on emotional tone (1-2 sentences)
2. Commonsense Knowledge: Relevant keywords (5-7 keywords)

Output as JSON with keys: "scene" and "knowledge" (knowledge is a list)
Output ONLY valid JSON, no markdown."""
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512,
            )
        )
        
        response_text = response.text.strip()
        
        # Clean markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        enrichment = json.loads(response_text)
        scene = enrichment.get('scene', 'Unable to determine')
        knowledge = enrichment.get('knowledge', ['unknown'])
        
        enriched_text = f"{utterance} Scene: {scene} Knowledge: {', '.join(knowledge)}"
        
        print(f"Scene: {scene}")
        print(f"Knowledge: {', '.join(knowledge)}")
        
        # Encode
        encoded = tokenizer(enriched_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            embedding = model(input_ids, attention_mask).cpu().numpy()
        
        # k-NN
        distances = cdist(embedding, train_embeddings, metric='euclidean').flatten()
        k_nearest_indices = np.argsort(distances)[:5]
        k_nearest_labels = [train_labels_str[i] for i in k_nearest_indices]
        
        label_counts = Counter(k_nearest_labels)
        predicted_emotion = label_counts.most_common(1)[0][0]
        predicted_pedagogical = EMOTION_TO_PEDAGOGICAL[predicted_emotion]
        
        print(f"5-NN: {k_nearest_labels}")
        print(f"→ Emotion: {predicted_emotion.upper()} → Pedagogical: {predicted_pedagogical.upper()}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("-"*80)

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)