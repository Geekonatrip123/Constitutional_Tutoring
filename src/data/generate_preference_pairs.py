"""
Generate Preference Pairs for DPO Training - Task 2.4.1
Creates pairs of deliberations for the same context with quality labels.
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


# Pedagogical States
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

# Math topics for Algebra (Grades 5-12) - COMPREHENSIVE
MATH_TOPICS = [
    # Basic Algebra
    "evaluating expressions with variables",
    "order of operations (PEMDAS/BODMAS)",
    "combining like terms",
    "using the distributive property",
    "writing expressions from word problems",
    
    # Equations & Inequalities
    "solving one-step equations",
    "solving multi-step equations",
    "equations with variables on both sides",
    "solving literal equations (solving for a variable)",
    "solving absolute value equations",
    "graphing inequalities on number lines",
    "solving compound inequalities",
    "solving quadratic inequalities",
    
    # Linear Functions & Graphs
    "plotting points on coordinate plane",
    "finding slope from two points",
    "finding slope from a graph",
    "understanding slope-intercept form (y = mx + b)",
    "converting between slope-intercept and standard form",
    "writing equations of lines",
    "graphing linear equations",
    "finding x- and y-intercepts",
    "understanding parallel and perpendicular lines",
    "interpreting graphs of real-world situations",
    
    # Systems of Equations
    "solving systems by graphing",
    "solving systems by substitution",
    "solving systems by elimination",
    "systems with no solution or infinite solutions",
    "word problems with systems of equations",
    
    # Exponents & Radicals
    "laws of exponents (product, quotient, power rules)",
    "negative and zero exponents",
    "fractional exponents",
    "simplifying radical expressions",
    "adding and subtracting radicals",
    "multiplying radicals",
    "rationalizing denominators",
    "solving radical equations",
    
    # Polynomials
    "adding and subtracting polynomials",
    "multiplying polynomials",
    "dividing polynomials",
    "factoring out greatest common factor (GCF)",
    "factoring trinomials (x² + bx + c)",
    "factoring difference of squares",
    "factoring perfect square trinomials",
    "factoring by grouping",
    
    # Quadratics
    "solving quadratics by factoring",
    "solving quadratics using square roots",
    "completing the square",
    "using the quadratic formula",
    "graphing parabolas (vertex form)",
    "finding vertex, axis of symmetry, and intercepts",
    "word problems involving quadratic models",
    
    # Rational Expressions & Equations
    "simplifying rational expressions",
    "multiplying and dividing rational expressions",
    "adding and subtracting rational expressions",
    "solving rational equations",
    "direct and inverse variation",
    "proportion word problems",
    
    # Functions
    "understanding function notation f(x)",
    "evaluating functions",
    "domain and range of functions",
    "composition of functions",
    "inverse functions",
    "piecewise functions",
    "transformations of functions (shifts, stretches)",
    
    # Sequences & Series
    "arithmetic sequences",
    "geometric sequences",
    "finding nth term",
    "summation notation",
    
    # Data & Statistics
    "mean, median, mode, range",
    "interpreting box plots and histograms",
    "scatter plots and correlation",
    "line of best fit",
    "two-way tables and conditional probability",
    
    # Word Problems & Applications
    "distance-rate-time problems",
    "mixture problems",
    "work rate problems",
    "percentage problems (markup, discount, interest)",
    "area and perimeter with variables",
    "modeling real-world situations with equations"
]

# State descriptions
STATE_DESCRIPTIONS = {
    "confused": "The student doesn't understand a concept, method, or strategy, or is stuck on how to proceed with the problem",
    "frustrated": "The student has tried multiple times, keeps making errors, feels overwhelmed, or is close to giving up",
    "correct": "The student solved the problem correctly, had an insight, made progress, or demonstrated understanding",
    "disengaged": "The student is bored, doesn't see the relevance, questions why they need to learn this, or has checked out mentally",
    "neutral": "The student is calmly asking for next steps, clarification, examples, or how to start a problem"
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


class PreferencePairGenerator:
    """Generate preference pairs for DPO training."""
    
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self._configure_current_key()
        
        logger.info(f"Generator initialized with {len(api_keys)} API keys")
    
    def _configure_current_key(self):
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.client = genai.GenerativeModel(self.model_name)
        logger.info(f"Using API key #{self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()
    
    def create_batch_prompt(self, batch_items: List[Dict[str, str]]) -> str:
        """Create prompt for generating preference pairs."""
        
        batch_descriptions = []
        for idx, item in enumerate(batch_items):
            state = item['student_state']
            action = item['candidate_action']
            math_topic = random.choice(MATH_TOPICS)
            state_desc = STATE_DESCRIPTIONS[state]
            
            batch_descriptions.append(f"""
Item {idx}:
- Student State: {state} ({state_desc})
- Candidate Action: {action}
- Math Topic: {math_topic}
""")
        
        batch_text = "\n".join(batch_descriptions)
        
        prompt = f"""You are a pedagogical expert designing an AI tutor for MATHEMATICS (Algebra, Grades 5-12).

PEDAGOGICAL PRINCIPLES:
{PEDAGOGICAL_PRINCIPLES}

TASK:
For each item below, generate:
1. A SPECIFIC math problem with actual numbers/expressions
2. A REALISTIC student utterance showing them working on that problem
3. TWO different deliberations (one high-quality, one lower-quality) explaining why the candidate action is appropriate

The TWO deliberations should:
- Address the same context but with different reasoning quality
- High-quality: References 2+ principles, clear reasoning, pedagogically sound
- Lower-quality: References 1 principle or has weaker reasoning, still acceptable but not as strong

CRITICAL REQUIREMENTS:

**FOR THE MATH PROBLEM:**
- Include ACTUAL numbers, equations, graphs, or expressions
- Cover diverse algebra topics (not just equations!)
- Appropriate for Grades 5-12
- Be specific and concrete

**FOR THE STUDENT UTTERANCE:**
- Show the student working on that specific problem
- Reference actual numbers/expressions from the problem
- Match the emotional state naturally
- Sound like a real student (ages 11-18)

**FOR THE TWO DELIBERATIONS:**
- Both should be valid but one clearly better
- High-quality: 2-3 sentences, references 2+ principles by number, excellent reasoning
- Lower-quality: 2-3 sentences, references 1 principle or has weaker reasoning
- Both must be complete sentences with proper punctuation

ITEMS TO GENERATE:
{batch_text}

OUTPUT FORMAT (JSON):
{{
  "item_0": {{
    "problem": "Simplify: 3(x + 4) - 2(x - 1)",
    "utterance": "I distributed and got 3x + 12 - 2x - 2. Now do I combine like terms to get x + 10?",
    "deliberation_high": "Since the student correctly distributed both terms but made a sign error with -2(x-1), I should provide_hint to guide them toward recognizing that -2 times -1 equals +2, not -2. This aligns with Principle 1 (Foster Constructivism) by helping them discover their error rather than just telling them. It also manages Principle 2 (Cognitive Load) by focusing on the specific misconception without overwhelming them with all the rules of distribution at once.",
    "score_high": 0.92,
    "deliberation_low": "The student made a sign error in distribution, so I should provide_hint to help them see that -2(-1) = +2. This follows Principle 1 (Constructivism) by guiding discovery of the error.",
    "score_low": 0.78
  }},
  "item_1": {{
    "problem": "Find the slope of the line passing through (2, 5) and (6, 13)",
    "utterance": "I used the formula and got (13-5)/(6-2) = 8/4 = 2. Is the slope 2?",
    "deliberation_high": "Given that the student correctly applied the slope formula and arrived at the right answer, I should give_positive_feedback to reinforce their successful use of the formula. This aligns with Principle 5 (Foster Positive Affect) by validating their correct work. Following up with Principle 4 (Promote Metacognition), I can ask them to explain what the slope of 2 means in context, deepening their understanding of rate of change.",
    "score_high": 0.94,
    "deliberation_low": "The student got the correct answer using the slope formula, so I should give_positive_feedback to acknowledge their work. This supports Principle 5 (Positive Affect) by validating effort.",
    "score_low": 0.81
  }}
}}

CRITICAL:
- Output ONLY valid JSON (no markdown, no extra text)
- Every problem must have SPECIFIC numbers/expressions
- Every utterance must reference those SPECIFIC numbers
- Both deliberations must be complete sentences
- High-quality should clearly be better than low-quality
- Make problems DIVERSE across algebra topics
"""
        return prompt
    
    def generate_batch_pairs(
        self,
        batch_items: List[Dict[str, str]],
        retry_attempts: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate batch of preference pairs."""
        
        batch_prompt = self.create_batch_prompt(batch_items)
        
        for attempt in range(retry_attempts):
            try:
                response = self.client.generate_content(
                    batch_prompt,
                    generation_config={
                        'temperature': 0.9,
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
                        delib_high = result_data.get('deliberation_high', '').strip()
                        delib_low = result_data.get('deliberation_low', '').strip()
                        score_high = result_data.get('score_high', 0.9)
                        score_low = result_data.get('score_low', 0.7)
                        
                        # Validate completeness
                        if (problem and utterance and delib_high and delib_low and
                            delib_high[-1] in ['.', '!', '?'] and 
                            delib_low[-1] in ['.', '!', '?'] and
                            score_high > score_low):
                            
                            results.append({
                                'student_state': item['student_state'],
                                'candidate_action': item['candidate_action'],
                                'math_problem': problem,
                                'student_utterance': utterance,
                                'chosen': delib_high,
                                'rejected': delib_low,
                                'score_chosen': float(score_high),
                                'score_rejected': float(score_low)
                            })
                        else:
                            logger.warning(f"{key} incomplete or invalid")
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
        target_total: int = 2000,
        batch_size: int = 10,
        rate_limit_delay: float = 3.0,
        resume: bool = True
    ):
        """Generate complete preference pairs dataset."""
        
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Track existing
        existing_count = 0
        if resume and output_path.exists():
            with open(output_path, 'r') as f:
                for line in f:
                    existing_count += 1
            logger.info(f"RESUMING: Found {existing_count} existing pairs")
        
        remaining = target_total - existing_count
        
        if remaining <= 0:
            logger.info("Dataset already complete!")
            return
        
        logger.info(f"Target: {target_total} preference pairs")
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
            
            with tqdm(total=remaining, desc="Generating preference pairs") as pbar:
                for i in range(0, len(all_items), batch_size):
                    batch = all_items[i:i+batch_size]
                    
                    self._rotate_key()
                    batch_count += 1
                    
                    results = self.generate_batch_pairs(batch)
                    
                    successful = 0
                    for result in results:
                        if result and result['chosen']:
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
        "AIzaSyBSktHO98I4-OpQZ839tSGARskV37uV1rI",
        "AIzaSyCoQpNeDZEkz3RcotbQs9NdOC_ZfCC8I3A",
        "AIzaSyBYYHNx2tmChCwqsMXIBPtajE01CLs9Tkw",
        "AIzaSyCULlSpSLFjN7pn1ijAPEG7oR6hqPjRNvY"
    ]
    MODEL_NAME = "gemini-2.5-flash"
    
    OUTPUT_DIR = Path("./data/processed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "preference_pairs.jsonl"
    
    generator = PreferencePairGenerator(
        api_keys=API_KEYS,
        model_name=MODEL_NAME
    )
    
    logger.info("=" * 80)
    logger.info("GENERATING PREFERENCE PAIRS FOR DPO TRAINING")
    logger.info("=" * 80)
    
    generator.generate_dataset(
        output_jsonl=str(OUTPUT_FILE),
        target_total=2000,  # 2K preference pairs
        batch_size=10,  # 10 pairs per call
        rate_limit_delay=3.0,
        resume=True
    )
    
    logger.info("=" * 80)
    logger.info("COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()