"""
COSIKE Enrichment for DailyDialog Dataset
Adds scene descriptions and commonsense knowledge using Gemini API.
Processes in batches to optimize API usage.
"""

import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COSIKEEnricher:
    """
    Enriches DailyDialog utterances with scene descriptions and commonsense knowledge.
    Uses Gemini API with batch processing.
    """
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.0-flash"):
        """
        Initialize COSIKE enricher.
        
        Args:
            api_key: Gemini API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"COSIKE Enricher initialized with model: {model_name}")
    
    def create_batch_prompt(self, utterances: List[str]) -> str:
        """
        Create a batch prompt for multiple utterances.
        
        Args:
            utterances: List of utterance texts
            
        Returns:
            Formatted batch prompt
        """
        prompt = """You are an expert in analyzing dialogues and emotional contexts.

For each of the following utterances in a conversational context, generate:
1. Scene Description: A brief description of the conversational context, with specific emphasis on the emotional tone and any feelings being expressed by the speaker (1-2 sentences)
2. Commonsense Knowledge: Relevant keywords or concepts related to the utterance (5-7 keywords)

Output as a JSON array where each element corresponds to an utterance, in the same order as input.

Format:
[
  {
    "utterance_id": 0,
    "scene": "scene description here",
    "knowledge": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
  },
  ...
]

UTTERANCES:
"""
        
        for idx, utterance in enumerate(utterances):
            prompt += f"\n{idx}. \"{utterance}\""
        
        prompt += "\n\nOutput valid JSON array only (no markdown, no extra text):"
        
        return prompt
    
    def enrich_batch(self, utterances: List[str], retry_attempts: int = 3) -> List[Dict[str, Any]]:
        """
        Enrich a batch of utterances using Gemini API.
        
        Args:
            utterances: List of utterance texts
            retry_attempts: Number of retry attempts on failure
            
        Returns:
            List of enrichment results (scene + knowledge for each utterance)
        """
        prompt = self.create_batch_prompt(utterances)
        
        for attempt in range(retry_attempts):
            try:
                # Generate with Gemini
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                    )
                )
                
                # Extract text
                response_text = response.text.strip()
                
                # Clean markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                enrichments = json.loads(response_text)
                
                # Validate
                if not isinstance(enrichments, list):
                    raise ValueError("Response is not a JSON array")
                
                if len(enrichments) != len(utterances):
                    logger.warning(
                        f"Mismatch: Expected {len(utterances)} enrichments, got {len(enrichments)}"
                    )
                
                # Fill in missing enrichments with defaults
                while len(enrichments) < len(utterances):
                    enrichments.append({
                        "utterance_id": len(enrichments),
                        "scene": "Unable to determine scene",
                        "knowledge": ["unknown"]
                    })
                
                logger.debug(f"Successfully enriched batch of {len(utterances)} utterances")
                return enrichments
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error (attempt {attempt + 1}/{retry_attempts}): {e}")
                logger.warning(f"Response text: {response_text[:200]}...")
                
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Return default enrichments on final failure
                    logger.error("Failed to parse JSON after all attempts, using defaults")
                    return [
                        {
                            "utterance_id": i,
                            "scene": "Unable to determine scene",
                            "knowledge": ["unknown"]
                        }
                        for i in range(len(utterances))
                    ]
            
            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}/{retry_attempts}): {e}")
                
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed after all attempts, using defaults")
                    return [
                        {
                            "utterance_id": i,
                            "scene": "Unable to determine scene",
                            "knowledge": ["unknown"]
                        }
                        for i in range(len(utterances))
                    ]
    
    def parse_space_separated_list(self, text: str) -> List[int]:
        """
        Parse space-separated list format from CSV.
        
        Args:
            text: String like "[3 4 2 2 2 3 4 1 3 4]"
            
        Returns:
            List of integers
        """
        # Remove brackets and split by spaces
        text = text.strip()
        if text.startswith('['):
            text = text[1:]
        if text.endswith(']'):
            text = text[:-1]
        
        # Split and convert to integers
        return [int(x) for x in text.split() if x.strip()]
    
    def enrich_dataset(
        self,
        input_csv: str,
        output_jsonl: str,
        batch_size: int = 10,
        max_samples: int = None,
        save_every: int = 100
    ):
        """
        Enrich entire DailyDialog dataset.
        
        Args:
            input_csv: Path to input CSV file (train_daily.csv, validation_daily.csv, or test_daily.csv)
            output_jsonl: Path to output JSONL file
            batch_size: Number of utterances to process per API call
            max_samples: Maximum number of samples to process (None = all)
            save_every: Save progress every N samples
        """
        logger.info(f"Loading dataset from: {input_csv}")
        
        # Load CSV
        df = pd.read_csv(input_csv)
        
        # Limit samples if specified
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Total dialogues to process: {len(df)}")
        
        # Prepare output file
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open output file
        enriched_count = 0
        
        with open(output_path, 'w') as f_out:
            # Process each dialogue
            for dialogue_idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching dialogues"):
                # Parse dialogue
                try:
                    dialog_raw = eval(row['dialog'])  # Returns a list with 1 element
                    
                    # The single element contains all utterances separated by '\n '
                    if isinstance(dialog_raw, list) and len(dialog_raw) == 1:
                        # Split on newline to get individual utterances
                        dialog = [utt.strip() for utt in dialog_raw[0].split('  ') if utt.strip()]  # Double space!
                    else:
                        logger.warning(f"Unexpected dialog format at index {dialogue_idx}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Failed to parse dialog at index {dialogue_idx}: {e}")
                    continue
                
                # Parse acts and emotions (space-separated format)
                acts = self.parse_space_separated_list(row['act'])
                emotions = self.parse_space_separated_list(row['emotion'])
                
                # Validate lengths match
                if not (len(dialog) == len(acts) == len(emotions)):
                    logger.warning(
                        f"Length mismatch in dialogue {dialogue_idx}: "
                        f"dialog={len(dialog)}, acts={len(acts)}, emotions={len(emotions)}"
                    )
                    # Skip this dialogue
                    continue
                
                # Process utterances in batches
                enriched_utterances = []
                
                for i in range(0, len(dialog), batch_size):
                    batch_utterances = dialog[i:i+batch_size]
                    
                    # Enrich batch
                    enrichments = self.enrich_batch(batch_utterances)
                    enriched_utterances.extend(enrichments)
                    
                    # Rate limiting: small delay between batches
                    time.sleep(0.5)
                
                # Combine with original data
                dialogue_data = {
                    'dialogue_id': dialogue_idx,
                    'utterances': []
                }
                
                for i, (utterance, act, emotion, enrichment) in enumerate(
                    zip(dialog, acts, emotions, enriched_utterances)
                ):
                    utterance_data = {
                        'turn': i,
                        'text': utterance,
                        'act': act,
                        'emotion': emotion,  # Original DailyDialog emotion (0-6)
                        'scene': enrichment.get('scene', 'Unable to determine scene'),
                        'knowledge': enrichment.get('knowledge', ['unknown'])
                    }
                    dialogue_data['utterances'].append(utterance_data)
                
                # Write to JSONL
                f_out.write(json.dumps(dialogue_data) + '\n')
                enriched_count += 1
                
                # Periodic save/flush
                if enriched_count % save_every == 0:
                    f_out.flush()
                    logger.info(f"Progress: {enriched_count}/{len(df)} dialogues enriched")
        
        logger.info(f"Enrichment complete! Saved to: {output_path}")
        logger.info(f"Total dialogues enriched: {enriched_count}")


def main():
    """Main execution function."""
    
    # Configuration
    API_KEY = "AIzaSyAsl7Dvet04WIc45yL-1GsHAfy1DScaGEQ"
    MODEL_NAME = "models/gemini-2.0-flash"
    
    # Paths
    DATA_DIR = Path("./data/raw/dailydialog")
    OUTPUT_DIR = Path("./data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize enricher
    enricher = COSIKEEnricher(api_key=API_KEY, model_name=MODEL_NAME)
    
    # TEST RUN FIRST - Only 10 dialogues
    logger.info("=" * 80)
    logger.info("TEST RUN - Processing 10 dialogues")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "train_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "test_sample_enriched.jsonl"),
        batch_size=10,
        max_samples=10,  # Only 10 for testing
        save_every=5
    )
    
    # Uncomment below to run full enrichment after testing
  
    # Enrich train set
    logger.info("=" * 80)
    logger.info("ENRICHING TRAINING SET")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "train_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "train_dailydialog_enriched.jsonl"),
        batch_size=10,
        max_samples=None,
        save_every=100
    )
    
    # Enrich validation set
    logger.info("=" * 80)
    logger.info("ENRICHING VALIDATION SET")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "validation_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "validation_dailydialog_enriched.jsonl"),
        batch_size=10,
        max_samples=None,
        save_every=50
    )
    
    # Enrich test set
    logger.info("=" * 80)
    logger.info("ENRICHING TEST SET")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "test_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "test_dailydialog_enriched.jsonl"),
        batch_size=10,
        max_samples=None,
        save_every=50
    )
   
    
    logger.info("=" * 80)
    logger.info("ENRICHMENT COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()