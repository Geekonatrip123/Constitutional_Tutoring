"""
Generate Synthetic Deliberations Dataset for Math/Algebra (Grades 5-12)
Uses Gemini API with key rotation and batch generation.
Generates REAL math problems and corresponding deliberations.
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

# Math problem types (Grades 5-12)
PROBLEM_TYPES = [
    "solving linear equations (e.g., 3x + 5 = 14)",
    "simplifying algebraic expressions (e.g., 2(x + 3) - 5x)",
    "factoring quadratics (e.g., x² + 5x + 6)",
    "solving equations with fractions (e.g., x/3 + 2 = 5)",
    "graphing linear functions (e.g., y = 2x - 3)",
    "finding slope from two points",
    "applying distributive property (e.g., 4(2x - 3))",
    "solving systems of equations",
    "working with exponents (e.g., simplifying x³ · x²)",
    "word problems involving equations",
    "evaluating functions (e.g., f(x) = 2x + 1, find f(3))",
    "simplifying radicals (e.g., √50)",
    "polynomial operations (e.g., (x + 2)(x - 3))",
    "solving inequalities (e.g., 2x - 5 < 7)"
]

# State descriptions
STATE_DESCRIPTIONS = {
    "confused": "The student is confused, doesn't understand the concept or method, or doesn't know how to proceed with the problem",
    "frustrated": "The student is frustrated, has tried multiple times and keeps making errors, or feels overwhelmed by the problem",
    "correct": "The student got the right answer, had an insight, or correctly applied a method",
    "disengaged": "The student is bored, doesn't see the relevance, or wants to give up on the problem",
    "neutral": "The student is calmly asking for next steps, clarification, or how to approach the problem"
}

# Pedagogical Principles
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
    Generates synthetic deliberations with real math problems.
    """

    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self._configure_current_key()
        
        logger.info(f"Generator initialized with {len(api_keys)} API keys")
        logger.info(f"Model: {model_name}")

    def _configure_current_key(self):
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f"Using API key #{self.current_key_index + 1}/{len(self.api_keys)}")

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()

    def create_batch_prompt(self, batch_items: List[Dict[str, str]]) -> str:
        """Create prompt for batch generation with REAL math problems."""
        
        batch_descriptions = []
        for idx, item in enumerate(batch_items):
            state = item['student_state']
            action = item['candidate_action']
            problem_type = random.choice(PROBLEM_TYPES)
            state_desc = STATE_DESCRIPTIONS[state]
            
            batch_descriptions.append(f"""
Item {idx}:
- Student State: {state} ({state_desc})
- Candidate Action: {action}
- Problem Type: {problem_type}
""")

        batch_text = "\n".join(batch_descriptions)

        prompt = f"""You are a pedagogical expert designing an AI tutor for MATHEMATICS (Algebra, Grades 5-12).

PEDAGOGICAL PRINCIPLES:
{PEDAGOGICAL_PRINCIPLES}

TASK:
For each item below:
1. Generate a SPECIFIC math problem with actual numbers/equations
2. Generate a REALISTIC student utterance showing them working on that specific problem
3. Generate a deliberation explaining WHY the candidate action is pedagogically appropriate

CRITICAL REQUIREMENTS:

**FOR THE MATH PROBLEM:**
- Include ACTUAL equations, numbers, or expressions (e.g., "3x + 7 = 16" not "a linear equation")
- Make it appropriate for middle/high school (Grades 5-12)
- Vary difficulty and complexity
- Be specific and concrete

**FOR THE STUDENT UTTERANCE:**
- Show the student WORKING ON that specific problem
- Reference the actual numbers/equation from the problem
- Match the emotional state naturally
- Sound like a real student (ages 11-18)
- Examples:
  * confused: "I got 2x = 6 but I don't know what to do next"
  * frustrated: "I keep getting x = -2 but the book says x = 3!"
  * correct: "Oh! So x = 5 because 3(5) + 7 = 22?"
  * disengaged: "Why do I need to solve for x in 2x - 8 = 4?"
  * neutral: "For 5x + 3 = 18, do I subtract 3 first?"

**FOR THE DELIBERATION:**
- Reference AT LEAST 2 pedagogical principles by number
- Explain why this action is appropriate for THIS specific problem/situation
- Mention the mathematical concept involved
- Be 2-3 COMPLETE sentences
- Focus on learning outcomes

ITEMS TO GENERATE:
{batch_text}

OUTPUT FORMAT (JSON):
{{
  "item_0": {{
    "problem": "Solve for x: 3x + 7 = 22",
    "utterance": "I subtracted 7 from both sides and got 3x = 15, but now I'm stuck. What do I do with the 3?",
    "deliberation": "Since the student correctly performed the first step but is now confused about isolating x, I should ask_socratic_question to guide them toward dividing both sides by 3. This aligns with Principle 1 (Foster Constructivism) by helping them discover the division step rather than telling them directly. It also respects Principle 2 (Manage Cognitive Load) by building on their correct work rather than introducing new concepts.",
    "score": 0.92
  }},
  "item_1": {{
    "problem": "Simplify: 2(x + 4) - 3x",
    "utterance": "I distributed and got 2x + 8 - 3x. Now do I just get -x + 8?",
    "deliberation": "Given that the student correctly distributed but needs confirmation about combining like terms, I should give_positive_feedback first to validate their distribution work. This follows Principle 5 (Foster Positive Affect) by acknowledging their correct steps. Then I can ask_for_explanation using Principle 4 (Promote Metacognition) to have them articulate why 2x - 3x = -x, deepening their understanding of combining terms.",
    "score": 0.89
  }}
}}

CRITICAL:
- Every problem must have SPECIFIC numbers/equations
- Every utterance must reference those SPECIFIC numbers
- Output ONLY valid JSON (no markdown, no extra text)
- Make problems DIVERSE across items
- Vary difficulty levels naturally
"""
        return prompt

    def generate_batch_deliberations(
        self,
        batch_items: List[Dict[str, str]],
        retry_attempts: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate batch with real math problems."""
        
        batch_prompt = self.create_batch_prompt(batch_items)

        for attempt in range(retry_attempts):
            try:
                response = self.client.generate_content(
                    batch_prompt,
                    generation_config={
                        'temperature': 0.9,  # High for diversity
                        'max_output_tokens': 32000,
                    },
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                )

                response_text = response.text.strip()

                # Clean markdown
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                # Parse JSON
                batch_results = json.loads(response_text)

                # Convert to output format
                results = []
                for idx, item in enumerate(batch_items):
                    key = f"item_{idx}"
                    if key in batch_results:
                        result_data = batch_results[key]
                        problem = result_data.get('problem', '').strip()
                        utterance = result_data.get('utterance', '').strip()
                        deliberation = result_data.get('deliberation', '').strip()

                        # Validate completeness
                        if (problem and utterance and deliberation and 
                            deliberation[-1] in ['.', '!', '?']):
                            results.append({
                                'student_state': item['student_state'],
                                'candidate_action': item['candidate_action'],
                                'math_problem': problem,
                                'student_utterance': utterance,
                                'deliberation_text': deliberation,
                                'alignment_score': float(result_data.get('score', 0.5))
                            })
                        else:
                            logger.warning(f"{key} incomplete")
                            results.append(None)
                    else:
                        logger.warning(f"Missing {key}")
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

                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(f"Rate limit hit, rotating key...")
                    self._rotate_key()
                    time.sleep(1)
                    continue

                logger.warning(f"Attempt {attempt + 1}/{retry_attempts} failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after all attempts")
                    return [None] * len(batch_items)

        return [None] * len(batch_items)

    def generate_dataset(
        self,
        output_jsonl: str,
        target_total: int = 5000,
        batch_size: int = 15,
        rate_limit_delay: float = 3.0,
        resume: bool = True
    ):
        """Generate complete dataset with real math problems."""
        
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Track existing
        existing_count = 0
        if resume and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    existing_count += 1
            logger.info(f"RESUMING: Found {existing_count} existing samples")

        remaining = target_total - existing_count

        if remaining <= 0:
            logger.info("Dataset already complete!")
            return

        logger.info(f"Target: {target_total}")
        logger.info(f"Existing: {existing_count}")
        logger.info(f"Remaining: {remaining}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Estimated time: ~{(remaining // batch_size) * rate_limit_delay / 60:.1f} minutes")

        # Generate random (state, action) pairs
        all_items = []
        for _ in range(remaining):
            state = random.choice(PEDAGOGICAL_STATES)
            action = random.choice(STATE_TO_ACTIONS[state])
            all_items.append({
                'student_state': state,
                'candidate_action': action
            })

        # Generate
        with open(output_path, 'a') as f:
            sample_count = existing_count
            batch_count = 0

            with tqdm(total=remaining, desc="Generating deliberations") as pbar:
                for i in range(0, len(all_items), batch_size):
                    batch = all_items[i:i+batch_size]

                    self._rotate_key()
                    batch_count += 1

                    results = self.generate_batch_deliberations(batch)

                    successful = 0
                    for result in results:
                        if result and result['deliberation_text']:
                            f.write(json.dumps(result) + '\n')
                            f.flush()
                            sample_count += 1
                            successful += 1
                            pbar.update(1)

                    if batch_count % 50 == 0:
                        logger.info(f"Progress: {sample_count}/{target_total} | Batches: {batch_count} | Success: {successful}/{len(batch)}")

                    time.sleep(rate_limit_delay)

        logger.info(f"Generation complete! Total: {sample_count}")


def main():
    """Main execution."""

    API_KEYS = [
        "AIzaSyAsl7Dvet04WIc45yL-1GsHAfy1DScaGEQ",
        "AIzaSyCTql2C66tkY3IPVMV0g0K_CHae_ZCL7FU",
        "AIzaSyCOJ2GlTMWvgbZBlD0DC38EPqy_NZ_dGoU",
        "AIzaSyBSktHO98I4-OpQZ839tSGARskV37uV1rI"
    ]
    MODEL_NAME = "gemini-2.5-flash"

    OUTPUT_DIR = Path("./data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_deliberations.jsonl"

    generator = SyntheticDeliberationGenerator(
        api_keys=API_KEYS,
        model_name=MODEL_NAME
    )

    logger.info("=" * 80)
    logger.info("GENERATING DELIBERATIONS WITH REAL MATH PROBLEMS")
    logger.info("=" * 80)

    generator.generate_dataset(
        output_jsonl=str(OUTPUT_FILE),
        target_total=5000,  # 5K deliberations with REAL problems
        batch_size=15,  # Smaller batch for real problems
        rate_limit_delay=3.0,
        resume=True
    )

    logger.info("=" * 80)
    logger.info("COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()