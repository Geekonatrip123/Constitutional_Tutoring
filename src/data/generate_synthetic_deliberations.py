"""
Generate Synthetic Deliberations Dataset for Math/Algebra (Grades 5-12)
Uses Gemini API with key rotation and batch generation.
Supports resume capability.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pedagogical States (from k-NN classifier output)
PEDAGOGICAL_STATES = ["confused", "frustrated", "correct", "disengaged", "neutral"]

# Actions mapped to each state
STATE_TO_ACTIONS = {
    "confused": [
        "ask_socratic_question",
        "provide_hint",
        "reframe_problem",
        "define_key_term",
        "simplify_problem",
        "check_for_understanding"
    ],
    "frustrated": [
        "validate_emotion",
        "simplify_problem",
        "provide_hint",
        "give_positive_feedback",
        "give_direct_answer"
    ],
    "correct": [
        "give_positive_feedback",
        "increase_difficulty",
        "ask_for_explanation",
        "check_for_understanding"
    ],
    "disengaged": [
        "validate_emotion",
        "reframe_problem",
        "show_worked_example",
        "give_positive_feedback"
    ],
    "neutral": [
        "ask_socratic_question",
        "check_for_understanding",
        "define_key_term",
        "provide_hint"
    ]
}

# Sample student utterances for each state (MATH-SPECIFIC, Grades 5-12)
STATE_UTTERANCES = {
    "confused": [
        "I don't understand what a variable is",
        "I'm confused about how to solve for x",
        "What does 'like terms' mean?",
        "I don't get how to use the distributive property",
        "How do I know when to add or multiply?",
        "I'm lost on this equation",
        "What's the difference between an expression and an equation?",
        "I don't understand why we need to balance both sides",
        "How do you simplify fractions with variables?",
        "I'm confused about negative exponents",
        "What does slope mean in this graph?",
        "I don't get how to factor this quadratic"
    ],
    "frustrated": [
        "I've tried this problem 5 times and keep getting it wrong!",
        "This algebra is impossible! I'll never understand it!",
        "I hate word problems, they make no sense!",
        "Why do I keep making the same mistake?",
        "I can't figure out where I went wrong",
        "This is too hard, I give up on this equation",
        "I'm so confused and frustrated with these fractions",
        "I've been stuck on this for 20 minutes!",
        "Every time I think I understand, I get the answer wrong",
        "I'm never going to pass this math test"
    ],
    "correct": [
        "I got x = 5, is that right?",
        "Oh! So I need to combine like terms first!",
        "I think the answer is 3x + 7",
        "I finally factored it: (x+3)(x-2)",
        "So the slope is 2/3?",
        "I see! You have to distribute first!",
        "The solution is x = -4, right?",
        "I solved it and got y = 2x + 1",
        "I understand now! First isolate the variable",
        "Is the answer 12?"
    ],
    "disengaged": [
        "When am I ever going to use algebra in real life?",
        "This is boring, can we do something else?",
        "Do I really have to solve all these equations?",
        "I don't care about finding x",
        "Math is pointless, I'm not going to use this",
        "Can we just skip to the answer?",
        "This is taking forever, I'm done",
        "I don't see why any of this matters"
    ],
    "neutral": [
        "Okay, what's the next step?",
        "How do I start this problem?",
        "Can you explain this example?",
        "What should I do first?",
        "Let me try solving this",
        "I'm ready for the next problem",
        "Can you show me how to do this type?",
        "What's the formula for this?"
    ]
}

# Math topics for context (Grades 5-12)
MATH_TOPICS = [
    "solving linear equations",
    "simplifying algebraic expressions",
    "factoring quadratics",
    "working with fractions and variables",
    "graphing linear functions",
    "understanding slope and y-intercept",
    "applying the distributive property",
    "solving systems of equations",
    "working with exponents and powers",
    "solving word problems with equations",
    "understanding functions",
    "simplifying radicals",
    "working with polynomials",
    "solving inequalities"
]

# Pedagogical Principles (from your framework)
PEDAGOGICAL_PRINCIPLES = """
1. Foster Constructivism & Scaffolding: Guide, don't just tell—but fade support as they learn.
2. Manage Cognitive Load: Don't overwhelm the student.
3. Maintain Desirable Difficulty: Keep the student in the 'zone'.
4. Promote Metacognition: Encourage reflection on thinking.
5. Foster Positive Affect & Validation: Validate feelings and encourage effort.
6. Maintain Factual Integrity: Be accurate; do not hallucinate.
"""


class SyntheticDeliberationGenerator:
    """
    Generates synthetic deliberations using Gemini API with key rotation.
    Supports resume capability and batch generation.
    """

    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        """
        Initialize generator with multiple API keys.

        Args:
            api_keys: List of Gemini API keys to rotate through
            model_name: Gemini model to use
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0

        # Initialize with first key
        self._configure_current_key()

        logger.info(f"Generator initialized with {len(api_keys)} API keys")
        logger.info(f"Model: {model_name}")

    def _configure_current_key(self):
        """Configure Gemini with current API key."""
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f"Using API key #{self.current_key_index + 1}/{len(self.api_keys)}")

    def _rotate_key(self):
        """Rotate to next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()

    def create_batch_prompt(
        self,
        batch_items: List[Dict[str, str]]
    ) -> str:
        """
        Create prompt for batch deliberation generation.

        Args:
            batch_items: List of dicts with student_state, candidate_action, student_utterance

        Returns:
            Formatted prompt for batch generation
        """
        # Build the batch request
        batch_descriptions = []
        for idx, item in enumerate(batch_items):
            math_topic = random.choice(MATH_TOPICS)
            batch_descriptions.append(f"""
Item {idx}:
- Student State: {item['student_state']}
- Student Utterance: "{item['student_utterance']}"
- Candidate Action: {item['candidate_action']}
- Math Topic: {math_topic}
""")

        batch_text = "\n".join(batch_descriptions)

        prompt = f"""You are a pedagogical expert designing an AI tutor for MATHEMATICS (Algebra, Grades 5-12).

PEDAGOGICAL PRINCIPLES:
{PEDAGOGICAL_PRINCIPLES}

TASK:
For each item below, generate a deliberation (2-3 COMPLETE sentences) that explains:
1. WHY this action is appropriate for this student state in the context of teaching math/algebra
2. Which pedagogical principles (reference by number) support this choice
3. What makes this better than alternative actions for helping the student learn math

REQUIREMENTS FOR EACH DELIBERATION:
- Reference AT LEAST 2 pedagogical principles explicitly (by number)
- Provide clear reasoning about the student's mathematical learning needs
- Explain the expected learning outcome in terms of mathematical understanding
- Keep the context focused on algebra/math concepts (grades 5-12)
- CRITICAL: Write 2-3 COMPLETE sentences. Do NOT truncate mid-sentence. Finish all thoughts fully.

ITEMS TO GENERATE DELIBERATIONS FOR:
{batch_text}

OUTPUT FORMAT:
Return a JSON object where each key is "item_0", "item_1", etc., and each value is an object with:
- "deliberation": the deliberation text (2-3 COMPLETE sentences with full punctuation)
- "score": alignment score from 0.0 to 1.0 based on principle adherence and reasoning quality

Example output format:
{{
  "item_0": {{
    "deliberation": "Given that the student is confused about variables, I should ask_socratic_question rather than give_direct_answer. This aligns with Principle 1 (Foster Constructivism) because guiding discovery builds deeper understanding than direct explanation. It also respects Principle 2 (Manage Cognitive Load) by breaking down the abstract concept through concrete examples step-by-step.",
    "score": 0.85
  }},
  "item_1": {{
    "deliberation": "Since the student shows frustration with equations, I should validate_emotion before proceeding with instruction. This follows Principle 5 (Foster Positive Affect) by acknowledging their struggle and normalizing difficulty. Then Principle 2 (Manage Cognitive Load) suggests simplifying the problem to rebuild confidence incrementally.",
    "score": 0.78
  }}
}}

CRITICAL:
- Output ONLY valid JSON. No markdown, no code blocks, no preamble, no explanation.
- Every deliberation must be 2-3 COMPLETE sentences ending with proper punctuation.
- Do NOT leave sentences incomplete or cut off mid-thought.
"""
        return prompt

    def generate_batch_deliberations(
        self,
        batch_items: List[Dict[str, str]],
        retry_attempts: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple deliberations in a single API call.

        Args:
            batch_items: List of dicts with student_state, candidate_action, student_utterance
            retry_attempts: Number of retry attempts on failure

        Returns:
            List of dictionaries with deliberation_text and alignment_score
        """
        batch_prompt = self.create_batch_prompt(batch_items)

        for attempt in range(retry_attempts):
            try:
                response = self.client.generate_content(
                    batch_prompt,
                    generation_config={
                        'temperature': 0.8,
                        'max_output_tokens': 32000,  # MASSIVE token limit for quality
                    },
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                )

                response_text = response.text.strip()

                # Clean markdown if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                # Parse JSON
                batch_results = json.loads(response_text)

                # Convert to list format
                results = []
                for idx, item in enumerate(batch_items):
                    key = f"item_{idx}"
                    if key in batch_results:
                        result_data = batch_results[key]
                        deliberation = result_data.get('deliberation', '').strip()

                        # Validate deliberation is complete (ends with punctuation)
                        if deliberation and deliberation[-1] in ['.', '!', '?']:
                            results.append({
                                'student_state': item['student_state'],
                                'candidate_action': item['candidate_action'],
                                'student_utterance': item['student_utterance'],
                                'deliberation_text': deliberation,
                                'alignment_score': float(result_data.get('score', 0.5))
                            })
                        else:
                            logger.warning(f"{key} deliberation incomplete: {deliberation[:50]}...")
                            results.append(None)
                    else:
                        logger.warning(f"Missing {key} in batch response")
                        results.append(None)

                return results

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}/{retry_attempts} - JSON parse failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to parse JSON after all attempts")
                    return [None] * len(batch_items)

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(f"Rate limit hit on key #{self.current_key_index + 1}, rotating to next key...")
                    self._rotate_key()
                    time.sleep(1)
                    # Retry with new key (don't count as failed attempt)
                    continue

                logger.warning(f"Attempt {attempt + 1}/{retry_attempts} failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after all attempts: {e}")
                    return [None] * len(batch_items)

        return [None] * len(batch_items)

    def generate_dataset(
        self,
        output_jsonl: str,
        samples_per_state_action: int = 15,
        batch_size: int = 20,
        rate_limit_delay: float = 3.0,
        resume: bool = True
    ):
        """
        Generate complete synthetic deliberations dataset with batching.

        Args:
            output_jsonl: Output JSONL file path
            samples_per_state_action: How many deliberations per (state, action) pair
            batch_size: Number of deliberations to generate per API call
            rate_limit_delay: Delay between API calls (seconds)
            resume: Whether to resume from existing file
        """
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Track what's already been generated with sample index
        existing_keys = set()
        existing_count = 0

        if resume and output_path.exists():
            # Count existing samples per (state, action, utterance)
            existing_combos = {}

            with open(output_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    combo = (data['student_state'], data['candidate_action'], data['student_utterance'])
                    existing_combos[combo] = existing_combos.get(combo, 0) + 1
                    existing_count += 1

            # Create keys for all existing samples
            for combo, count in existing_combos.items():
                for idx in range(count):
                    existing_keys.add((*combo, idx))

            logger.info(f"RESUMING: Found {existing_count} existing samples")
            logger.info(f"Unique combinations with samples: {len(existing_combos)}")

        # Generate all (state, action, utterance, sample_idx) combinations
        all_items = []
        for state in PEDAGOGICAL_STATES:
            # SKIP confused - we have enough!
            if state == "confused":
                continue

            for action in STATE_TO_ACTIONS[state]:
                for utterance in STATE_UTTERANCES[state]:
                    for sample_idx in range(samples_per_state_action):
                        # Create unique key with sample index
                        key = (state, action, utterance, sample_idx)

                        # Skip if already generated
                        if key not in existing_keys:
                            all_items.append({
                                'student_state': state,
                                'candidate_action': action,
                                'student_utterance': utterance
                            })

        # RANDOMIZE the order!
        random.shuffle(all_items)

        # Calculate totals
        total_target = existing_count + len(all_items)

        logger.info(f"Target total samples: {total_target}")
        logger.info(f"Already generated: {existing_count}")
        logger.info(f"Remaining: {len(all_items)}")
        logger.info(f"Batch size: {batch_size} deliberations per API call")
        logger.info(f"Estimated API calls: {len(all_items) // batch_size}")
        logger.info(f"With {len(self.api_keys)} keys rotating every batch, estimated time: ~{(len(all_items) // batch_size) * rate_limit_delay / 60:.1f} minutes")

        if len(all_items) == 0:
            logger.info("Dataset already complete!")
            return

        # Open file in APPEND mode
        with open(output_path, 'a') as f:
            sample_count = existing_count
            batch_count = 0

            # Process in batches
            with tqdm(total=len(all_items), desc="Generating deliberations") as pbar:
                for i in range(0, len(all_items), batch_size):
                    batch = all_items[i:i+batch_size]

                    # Rotate key EVERY batch for even distribution
                    self._rotate_key()
                    batch_count += 1

                    # Generate batch
                    results = self.generate_batch_deliberations(batch)

                    # Write successful results
                    successful = 0
                    for result in results:
                        if result and result['deliberation_text']:
                            f.write(json.dumps(result) + '\n')
                            f.flush()
                            sample_count += 1
                            successful += 1
                            pbar.update(1)

                    # Progress log every 50 batches
                    if batch_count % 50 == 0:
                        logger.info(f"Progress: {sample_count}/{total_target} samples | Batches: {batch_count} | Success rate: {successful}/{len(batch)} | Key: #{self.current_key_index + 1}")

                    # Rate limiting
                    time.sleep(rate_limit_delay)

        logger.info(f"Dataset generation complete!")
        logger.info(f"Total samples: {sample_count}")
        logger.info(f"Saved to: {output_path}")


def main():
    """Main execution function."""

    # Configuration - ALL 4 API KEYS
    API_KEYS = [
        "AIzaSyBLc2tHZPkXGaKcO9EqiJKvu6HUxNV8yaw",
        "AIzaSyA7fYS--69i1gkDabUOv7IhktVUswDeUSo",
        "AIzaSyB9K2sxcDqJEqsbnnD2PZbcUtkhVy3c51o",
        "AIzaSyDCT77pa7v-rAOXWm6e_KLKU79PP9ZIuFQ"
    ]
    MODEL_NAME = "gemini-2.5-flash"

    # Output path
    OUTPUT_DIR = Path("./data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_deliberations.jsonl"

    # Initialize generator
    generator = SyntheticDeliberationGenerator(
        api_keys=API_KEYS,
        model_name=MODEL_NAME
    )

    # Generate dataset
    logger.info("=" * 80)
    logger.info("GENERATING SYNTHETIC DELIBERATIONS - MATH/ALGEBRA (GRADES 5-12)")
    logger.info("=" * 80)

    generator.generate_dataset(
        output_jsonl=str(OUTPUT_FILE),
        samples_per_state_action=15,  # 15 deliberations per combo
        batch_size=20,  # 20 deliberations per API call
        rate_limit_delay=3.0,  # 3 seconds between calls
        resume=True
    )

    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()