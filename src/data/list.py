"""
List all available Gemini models for your API key.
"""

import google.generativeai as genai

def list_available_models(api_key: str):
    """
    List all available models for the given API key.
    
    Args:
        api_key: Your Gemini API key
    """
    # Configure API
    genai.configure(api_key=api_key)
    
    print("\n" + "="*80)
    print("AVAILABLE GEMINI MODELS")
    print("="*80 + "\n")
    
    # List all models
    models = genai.list_models()
    
    generation_models = []
    embedding_models = []
    other_models = []
    
    for model in models:
        model_info = {
            'name': model.name,
            'display_name': model.display_name,
            'description': model.description,
            'supported_methods': model.supported_generation_methods
        }
        
        # Categorize by capability
        if 'generateContent' in model.supported_generation_methods:
            generation_models.append(model_info)
        elif 'embedContent' in model.supported_generation_methods:
            embedding_models.append(model_info)
        else:
            other_models.append(model_info)
    
    # Print generation models (the ones we can use for text generation)
    print("📝 TEXT GENERATION MODELS:")
    print("-" * 80)
    for i, model in enumerate(generation_models, 1):
        print(f"\n{i}. {model['display_name']}")
        print(f"   Model ID: {model['name']}")
        print(f"   Description: {model['description']}")
        print(f"   Methods: {', '.join(model['supported_methods'])}")
    
    # Print embedding models
    if embedding_models:
        print("\n\n🔢 EMBEDDING MODELS:")
        print("-" * 80)
        for i, model in enumerate(embedding_models, 1):
            print(f"\n{i}. {model['display_name']}")
            print(f"   Model ID: {model['name']}")
            print(f"   Description: {model['description']}")
    
    # Print other models
    if other_models:
        print("\n\n🔧 OTHER MODELS:")
        print("-" * 80)
        for i, model in enumerate(other_models, 1):
            print(f"\n{i}. {model['display_name']}")
            print(f"   Model ID: {model['name']}")
            print(f"   Methods: {', '.join(model['supported_methods'])}")
    
    print("\n" + "="*80)
    print(f"TOTAL MODELS: {len(generation_models) + len(embedding_models) + len(other_models)}")
    print(f"  - Generation: {len(generation_models)}")
    print(f"  - Embedding: {len(embedding_models)}")
    print(f"  - Other: {len(other_models)}")
    print("="*80 + "\n")
    
    # Recommendations
    print("💡 RECOMMENDED FOR YOUR USE CASE:")
    print("-" * 80)
    recommended = [m for m in generation_models if 'gemini-1.5' in m['name'].lower() or 'gemini-2.0' in m['name'].lower()]
    
    if recommended:
        for model in recommended[:3]:  # Top 3 recommendations
            print(f"  ✓ {model['display_name']}")
            print(f"    Use: model_name='{model['name']}'")
    else:
        print("  Use the first generation model from the list above")
    
    print("\n")


if __name__ == "__main__":
    API_KEY = "AIzaSyAsl7Dvet04WIc45yL-1GsHAfy1DScaGEQ"
    list_available_models(API_KEY)