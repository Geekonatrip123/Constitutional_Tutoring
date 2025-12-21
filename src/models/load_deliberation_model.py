"""
Load DPO-trained Deliberation Generator with NF4 Quantization
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from pathlib import Path


def load_model(adapter_path: str):
    """
    Load Qwen3-8B with DPO LoRA adapters using NF4 4-bit quantization.
    
    Args:
        adapter_path: Path to downloaded LoRA adapters
    
    Returns:
        model, tokenizer
    """
    print("=" * 80)
    print("LOADING DELIBERATION GENERATOR MODEL")
    print("Using NF4 4-bit Quantization for Efficient Inference")
    print("=" * 80)
    
    # Load PEFT config
    print(f"\n1. Loading adapter config from: {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    BASE_MODEL = peft_config.base_model_name_or_path
    print(f"   Base model: {BASE_MODEL}")
    
    print(f"\n2. Configuring NF4 quantization...")
    print("   Model size: 16GB → 4GB (75% compression)")
    print("   Quality retention: ~97%")
    
    # NF4 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"\n3. Loading base model with quantization...")
    print("   This may take 2-3 minutes (first time only)...")
    
    # Load base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"\n4. Loading LoRA adapters from: {adapter_path}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )
    
    print("\n5. Loading tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set to eval mode
    model.eval()
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n✅ Model loaded successfully!")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   VRAM allocated: {allocated:.2f} GB")
        print(f"   VRAM reserved: {reserved:.2f} GB")
        print(f"   Quantization: NF4 4-bit")
    else:
        print(f"\n✅ Model loaded on CPU")
    
    print("=" * 80)
    
    return model, tokenizer


def generate_deliberation(
    model,
    tokenizer,
    student_state: str,
    math_problem: str,
    student_utterance: str,
    candidate_action: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Generate a pedagogical deliberation.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        student_state: Student's emotional/cognitive state
        math_problem: The specific math problem
        student_utterance: What the student said
        candidate_action: Tutor action to deliberate about
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.7 = balanced)
        top_p: Nucleus sampling threshold
    
    Returns:
        Generated deliberation text
    """
    # Create prompt
    prompt = f"""You are a pedagogical AI tutor for mathematics (Algebra, Grades 5-12).

STUDENT STATE: {student_state}
MATH PROBLEM: {math_problem}
STUDENT UTTERANCE: {student_utterance}
TUTOR ACTION: {candidate_action}

Generate a pedagogical deliberation (2-3 sentences) explaining WHY this action is appropriate. Reference specific pedagogical principles by number:
1. Foster Constructivism & Scaffolding
2. Manage Cognitive Load
3. Maintain Desirable Difficulty
4. Promote Metacognition
5. Foster Positive Affect & Validation
6. Maintain Factual Integrity

DELIBERATION:"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate
    print(f"\n🤖 Generating deliberation...")
    import time
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    elapsed = time.time() - start
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract deliberation (after "DELIBERATION:")
    if "DELIBERATION:" in full_output:
        deliberation = full_output.split("DELIBERATION:")[-1].strip()
    else:
        deliberation = full_output
    
    print(f"⏱️  Generation took {elapsed:.1f} seconds")
    
    return deliberation


def main():
    """Example usage with multiple test cases."""
    
    # Path to your downloaded LoRA adapters
    ADAPTER_PATH = r"C:\Users\Shlok\Downloads\final_model"
    
    print("\n🚀 Starting model loading...\n")
    
    # Load model (this will take 2-3 minutes first time)
    model, tokenizer = load_model(ADAPTER_PATH)
    
    # Example 1: Confused student
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Confused Student")
    print("=" * 80)
    print("\nContext:")
    print("  State: confused")
    print("  Problem: Solve for x: 3x + 7 = 22")
    print("  Student: 'I subtracted 7 and got 3x = 15, but now I'm stuck.'")
    print("  Action: ask_socratic_question")
    
    deliberation = generate_deliberation(
        model=model,
        tokenizer=tokenizer,
        student_state="confused",
        math_problem="Solve for x: 3x + 7 = 22",
        student_utterance="I subtracted 7 from both sides and got 3x = 15, but now I'm stuck. What do I do with the 3?",
        candidate_action="ask_socratic_question"
    )
    
    print(f"\n📝 Generated Deliberation:\n{deliberation}\n")
    
    # Example 2: Frustrated student
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Frustrated Student")
    print("=" * 80)
    print("\nContext:")
    print("  State: frustrated")
    print("  Problem: Factor: x² + 5x + 6")
    print("  Student: 'I keep trying but can't find two numbers that work!'")
    print("  Action: validate_emotion")
    
    deliberation = generate_deliberation(
        model=model,
        tokenizer=tokenizer,
        student_state="frustrated",
        math_problem="Factor: x² + 5x + 6",
        student_utterance="I keep trying to factor this but I can't find two numbers that work! This is so frustrating!",
        candidate_action="validate_emotion"
    )
    
    print(f"\n📝 Generated Deliberation:\n{deliberation}\n")
    
    # Example 3: Correct student
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Correct Student")
    print("=" * 80)
    print("\nContext:")
    print("  State: correct")
    print("  Problem: Find slope between (2,5) and (6,13)")
    print("  Student: 'I got (13-5)/(6-2) = 8/4 = 2. Is that right?'")
    print("  Action: give_positive_feedback")
    
    deliberation = generate_deliberation(
        model=model,
        tokenizer=tokenizer,
        student_state="correct",
        math_problem="Find the slope between points (2, 5) and (6, 13)",
        student_utterance="I used the formula and got (13-5)/(6-2) = 8/4 = 2. Is the slope 2?",
        candidate_action="give_positive_feedback"
    )
    
    print(f"\n📝 Generated Deliberation:\n{deliberation}\n")
    
    print("=" * 80)
    print("✅ ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\n💡 Tip: The model is now loaded and ready for more queries!")
    print("   You can import this script and call generate_deliberation() directly.")


if __name__ == "__main__":
    main()