"""
COSIKE Enrichment - Batching ENTIRE DIALOGUES (10 dialogues per API call)
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
    Enriches DailyDialog by batching ENTIRE DIALOGUES together.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):  # NO "models/" prefix
        """Initialize COSIKE enricher."""
        self.api_key = api_key
        self.model_name = model_name
        
        genai.configure(api_key=self.api_key)
        
        # Configure safety settings to avoid blocks
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings
        )
        
        logger.info(f"COSIKE Enricher initialized with model: {model_name}")
    
    def create_multi_dialogue_prompt(self, dialogues: List[List[str]]) -> str:
        """
        Create a prompt for MULTIPLE DIALOGUES at once.
        
        Args:
            dialogues: List of dialogues, where each dialogue is a list of utterances
            
        Returns:
            Formatted prompt
        """
        prompt = """You are an expert in analyzing dialogues and emotional contexts.

For each utterance in the following dialogues, generate:
1. Scene Description: A brief description of the conversational context, with specific emphasis on the emotional tone and any feelings being expressed (1-2 sentences)
2. Commonsense Knowledge: Relevant keywords or concepts (5-7 keywords)

Output as a JSON object where keys are "dialogue_0", "dialogue_1", etc.
Each dialogue contains an array of enrichments matching its utterances in order.

Format:
{
  "dialogue_0": [
    {"scene": "...", "knowledge": ["...", "..."]},
    {"scene": "...", "knowledge": ["...", "..."]}
  ],
  "dialogue_1": [
    {"scene": "...", "knowledge": ["...", "..."]},
    ...
  ],
  ...
}

DIALOGUES:
"""
        
        for dialogue_idx, dialogue in enumerate(dialogues):
            prompt += f"\n\n=== DIALOGUE {dialogue_idx} ===\n"
            for utt_idx, utterance in enumerate(dialogue):
                prompt += f"{utt_idx}. \"{utterance}\"\n"
        
        prompt += "\n\nOutput valid JSON object only (no markdown):"
        
        return prompt
    
    def enrich_dialogue_batch(
        self,
        dialogues: List[List[str]],
        retry_attempts: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Enrich multiple dialogues in a single API call.
        
        Args:
            dialogues: List of dialogues (each dialogue is a list of utterances)
            retry_attempts: Number of retry attempts
            
        Returns:
            List of enrichment lists (one per dialogue)
        """
        prompt = self.create_multi_dialogue_prompt(dialogues)
        
        for attempt in range(retry_attempts):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=25000,  # INCREASED - 10 dialogues need more tokens
                    )
                )
                
                response_text = response.text.strip()
                
                # Clean markdown
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                enrichments_dict = json.loads(response_text)
                
                # Extract enrichments for each dialogue
                result = []
                for dialogue_idx in range(len(dialogues)):
                    key = f"dialogue_{dialogue_idx}"
                    
                    if key in enrichments_dict:
                        dialogue_enrichments = enrichments_dict[key]
                    else:
                        # Fallback: create defaults
                        dialogue_enrichments = [
                            {"scene": "Unable to determine scene", "knowledge": ["unknown"]}
                            for _ in dialogues[dialogue_idx]
                        ]
                    
                    # Ensure correct length
                    while len(dialogue_enrichments) < len(dialogues[dialogue_idx]):
                        dialogue_enrichments.append({
                            "scene": "Unable to determine scene",
                            "knowledge": ["unknown"]
                        })
                    
                    result.append(dialogue_enrichments)
                
                logger.debug(f"Successfully enriched batch of {len(dialogues)} dialogues")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON error (attempt {attempt + 1}/{retry_attempts}): {e}")
                logger.warning(f"Response text (first 500 chars): {response_text[:500] if 'response_text' in locals() else 'N/A'}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Return defaults for all dialogues
                    logger.error("Failed to parse JSON, using defaults")
                    return [
                        [{"scene": "Unable to determine scene", "knowledge": ["unknown"]}
                         for _ in dialogue]
                        for dialogue in dialogues
                    ]
            
            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed after all attempts, using defaults")
                    return [
                        [{"scene": "Unable to determine scene", "knowledge": ["unknown"]}
                         for _ in dialogue]
                        for dialogue in dialogues
                    ]
    
    def parse_space_separated_list(self, text: str) -> List[int]:
        """Parse space-separated list."""
        text = text.strip()
        if text.startswith('['):
            text = text[1:]
        if text.endswith(']'):
            text = text[:-1]
        return [int(x) for x in text.split() if x.strip()]
    
    def enrich_dataset(
        self,
        input_csv: str,
        output_jsonl: str,
        dialogues_per_batch: int = 10,  # Process 10 dialogues per API call
        max_samples: int = None,
        save_every: int = 50,
        rate_limit_delay: float = 2.5,
        resume: bool = True
    ):
        """
        Enrich dataset by batching multiple dialogues per API call.
        
        Args:
            input_csv: Input CSV file
            output_jsonl: Output JSONL file
            dialogues_per_batch: Number of dialogues to process per API call (default: 10)
            max_samples: Maximum dialogues to process
            save_every: Save progress every N dialogues
            rate_limit_delay: Delay between API calls (seconds)
            resume: Resume from existing output file
        """
        logger.info(f"Loading dataset from: {input_csv}")
        
        # Load CSV
        df = pd.read_csv(input_csv)
        
        if max_samples:
            df = df.head(max_samples)
        
        # Check for existing output and resume
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        already_processed = set()
        if resume and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    dialogue = json.loads(line)
                    already_processed.add(dialogue['dialogue_id'])
            
            logger.info(f"RESUMING: Found {len(already_processed)} already processed dialogues")
        
        logger.info(f"Total dialogues: {len(df)}")
        logger.info(f"Remaining: {len(df) - len(already_processed)}")
        logger.info(f"Dialogues per API call: {dialogues_per_batch}")
        logger.info(f"Estimated API calls: {(len(df) - len(already_processed)) // dialogues_per_batch}")
        
        # Parse all dialogues first
        all_dialogues = []
        
        for dialogue_idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dialogues"):
            # Skip if already processed
            if dialogue_idx in already_processed:
                continue
            
            try:
                dialog_raw = eval(row['dialog'])
                
                if isinstance(dialog_raw, list) and len(dialog_raw) == 1:
                    dialog = [utt.strip() for utt in dialog_raw[0].split('  ') if utt.strip()]
                else:
                    logger.warning(f"Unexpected format at {dialogue_idx}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Parse error at {dialogue_idx}: {e}")
                continue
            
            acts = self.parse_space_separated_list(row['act'])
            emotions = self.parse_space_separated_list(row['emotion'])
            
            if not (len(dialog) == len(acts) == len(emotions)):
                logger.warning(f"Length mismatch at {dialogue_idx}")
                continue
            
            all_dialogues.append({
                'dialogue_idx': dialogue_idx,
                'utterances': dialog,
                'acts': acts,
                'emotions': emotions
            })
        
        logger.info(f"Loaded {len(all_dialogues)} dialogues to enrich")
        
        # Process in batches of N dialogues
        enriched_count = len(already_processed)
        total_api_calls = 0
        
        with open(output_path, 'a') as f_out:
            for i in tqdm(range(0, len(all_dialogues), dialogues_per_batch), desc="Enriching dialogue batches"):
                batch = all_dialogues[i:i+dialogues_per_batch]
                
                # Extract just the utterances for enrichment
                batch_utterances = [d['utterances'] for d in batch]
                
                # Enrich the entire batch
                batch_enrichments = self.enrich_dialogue_batch(batch_utterances)
                
                total_api_calls += 1
                
                # Write each dialogue to output
                for dialogue_data, enrichments in zip(batch, batch_enrichments):
                    dialogue_output = {
                        'dialogue_id': dialogue_data['dialogue_idx'],
                        'utterances': []
                    }
                    
                    for turn_idx, (utt, act, emotion, enrichment) in enumerate(zip(
                        dialogue_data['utterances'],
                        dialogue_data['acts'],
                        dialogue_data['emotions'],
                        enrichments
                    )):
                        dialogue_output['utterances'].append({
                            'turn': turn_idx,
                            'text': utt,
                            'act': act,
                            'emotion': emotion,
                            'scene': enrichment.get('scene', 'Unable to determine scene'),
                            'knowledge': enrichment.get('knowledge', ['unknown'])
                        })
                    
                    # Write to file
                    f_out.write(json.dumps(dialogue_output) + '\n')
                    enriched_count += 1
                
                # Periodic flush
                if enriched_count % save_every == 0:
                    f_out.flush()
                    logger.info(f"Progress: {enriched_count}/{len(df)} dialogues ({total_api_calls} API calls)")
                
                # Rate limiting
                time.sleep(rate_limit_delay)
        
        logger.info(f"Enrichment complete! Saved to: {output_path}")
        logger.info(f"Total dialogues enriched: {enriched_count}")
        logger.info(f"Total API calls made: {total_api_calls}")


def main():
    """Main execution function."""
    
    # Configuration
    API_KEY = "AIzaSyAsl7Dvet04WIc45yL-1GsHAfy1DScaGEQ"
    MODEL_NAME = "gemini-2.5-flash"  # NO "models/" prefix!
    
    # Paths
    DATA_DIR = Path("./data/raw/dailydialog")
    OUTPUT_DIR = Path("./data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize enricher
    enricher = COSIKEEnricher(api_key=API_KEY, model_name=MODEL_NAME)
    
    # Enrich train set
    logger.info("=" * 80)
    logger.info("ENRICHING TRAINING SET (10 dialogues per API call)")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "train_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "train_dailydialog_enriched.jsonl"),
        dialogues_per_batch=10,  # 10 dialogues per API call!
        max_samples=None,
        save_every=50,
        rate_limit_delay=2.5,  # 2.5s between calls = ~1400 calls/hour
        resume=True  # Continue from where you left off
    )
    # Enrich validation set (NEW)
    logger.info("=" * 80)
    logger.info("ENRICHING VALIDATION SET")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "validation_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "validation_dailydialog_enriched.jsonl"),
        dialogues_per_batch=10,
        max_samples=None,
        save_every=50,
        rate_limit_delay=2.5,
        resume=True
    )
    
    # Enrich test set (NEW)
    logger.info("=" * 80)
    logger.info("ENRICHING TEST SET")
    logger.info("=" * 80)
    enricher.enrich_dataset(
        input_csv=str(DATA_DIR / "test_daily.csv"),
        output_jsonl=str(OUTPUT_DIR / "test_dailydialog_enriched.jsonl"),
        dialogues_per_batch=10,
        max_samples=None,
        save_every=50,
        rate_limit_delay=2.5,
        resume=True
    )
    
    logger.info("=" * 80)
    logger.info("ENRICHMENT COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()