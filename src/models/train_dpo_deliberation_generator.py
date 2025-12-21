
"""
Train Deliberation Generator with DPO - Task 2.4.2
Fine-tunes Qwen3-8B using Direct Preference Optimization.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_preference_pairs(jsonl_path: str):
    """Load preference pairs from JSONL."""
    data = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            pair = json.loads(line)
            data.append({
                'prompt': create_prompt(pair),
                'chosen': pair['chosen'],
                'rejected': pair['rejected']
            })
    
    logger.info(f"Loaded {len(data)} preference pairs")
    return data


def create_prompt(pair):
    """Create input prompt for the model."""
    prompt = f"""You are a pedagogical AI tutor for mathematics (Algebra, Grades 5-12).

STUDENT STATE: {pair['student_state']}
MATH PROBLEM: {pair['math_problem']}
STUDENT UTTERANCE: {pair['student_utterance']}
TUTOR ACTION: {pair['candidate_action']}

Generate a pedagogical deliberation (2-3 sentences) explaining WHY this action is appropriate. Reference specific pedagogical principles by number:
1. Foster Constructivism & Scaffolding
2. Manage Cognitive Load
3. Maintain Desirable Difficulty
4. Promote Metacognition
5. Foster Positive Affect & Validation
6. Maintain Factual Integrity

DELIBERATION:"""
    
    return prompt


def prepare_dataset(data):
    """Prepare dataset for DPO training."""
    dataset = Dataset.from_list(data)
    logger.info(f"Dataset prepared: {len(dataset)} examples")
    return dataset


def main():
    """Main training pipeline."""
    
    # Paths
    DATA_FILE = Path("/root/preference_pairs.jsonl")
    OUTPUT_DIR = Path("/root/models/deliberation_generator_dpo")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("DPO TRAINING - DELIBERATION GENERATOR (Qwen3-8B)")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading preference pairs...")
    data = load_preference_pairs(DATA_FILE)
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Load model and tokenizer
    MODEL_NAME = "Qwen/Qwen3-8B"
    logger.info(f"Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Reference model (frozen copy for DPO)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA for efficient fine-tuning
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data)
    eval_dataset = prepare_dataset(val_data)
    
    # DPO Configuration (NEW API)
    dpo_config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        bf16=True,
        eval_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        beta=0.1,  # DPO beta parameter
        max_length=1024,
        max_prompt_length=512,
    )
    
    # Initialize DPO trainer (NEW API)
    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("STARTING DPO TRAINING")
    logger.info("=" * 80)
    
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
    
    logger.info("=" * 80)
    logger.info("DPO TRAINING COMPLETE!")
    logger.info(f"Model saved to: {OUTPUT_DIR / 'final_model'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
