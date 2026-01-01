"""
Knowledge Tracing Integration with LLM-based correctness judgment.
Tracks student knowledge state and calculates Inferred Mastery Velocity (IMV).
Uses Qwen3-8B for judgment.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMCorrectnessJudge:
    """
    Uses Qwen3-8B to judge if student response indicates understanding.
    Maintains full conversation memory.
    Supports sharing model with deliberation generator to save VRAM.
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        use_quantization: bool = True,
        model=None,
        tokenizer=None,
        use_shared_model: bool = False
    ):
        """
        Initialize LLM judge.
        
        Args:
            model_path: Path to Qwen model (Qwen/Qwen3-8B)
            device: Device to run on
            use_quantization: Use 4-bit quantization
            model: Pre-loaded model (for sharing with deliberation generator)
            tokenizer: Pre-loaded tokenizer (for sharing)
            use_shared_model: Whether to use shared model (saves VRAM)
        """
        self.device = device
        
        if use_shared_model and model is not None and tokenizer is not None:
            # Use shared model - no loading needed
            logger.info("Using shared DPO model for knowledge tracing")
            self.model = model
            self.tokenizer = tokenizer
            logger.info("✅ LLM judge initialized (shared model)")
        else:
            # Load new model (original behavior)
            logger.info(f"Loading LLM judge: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            if use_quantization:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            self.model.eval()
            logger.info("✅ LLM judge loaded successfully")
    
    def judge_correctness(
        self,
        problem: str,
        student_message: str,
        expected_state: str,
        full_conversation_history: List[Dict[str, str]]
    ) -> Tuple[bool, float, str]:
        """
        Judge if student response indicates understanding/correctness.
        Uses FULL conversation history (all turns).
        
        Args:
            problem: The math problem being solved
            student_message: Student's current message
            expected_state: Expected emotional state
            full_conversation_history: ALL previous turns with tutor responses
                Format: [{"role": "student/tutor", "message": "..."}]
        
        Returns:
            (is_correct, confidence_score, reasoning)
        """
        # Build full conversation context
        history_text = ""
        if full_conversation_history:
            history_lines = []
            for turn in full_conversation_history:
                role = "Student" if turn['role'] == 'student' else "Tutor"
                history_lines.append(f"{role}: {turn['message']}")
            history_text = "\n".join(history_lines)
            history_text = f"\n\nFull conversation so far:\n{history_text}"
        
        # Construct prompt - SIMPLIFIED to avoid <think> tags
        prompt = f"""Evaluate if this student message shows understanding of the math problem.

Problem: {problem}
Student message: "{student_message}"
Student state: {expected_state}{history_text}

Does this message show correct understanding or progress? Consider:
- Correct math reasoning
- Appropriate next steps
- Progress from confusion to clarity

Respond with ONLY valid JSON (no other text):
{{"correct": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        # Generate
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.1,  # Lower temperature for more structured output
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse JSON response
        try:
            import json
            import re
            
            # Clean response
            response = response.strip()
            
            # Remove <think> tags if present
            if "<think>" in response:
                # Extract everything after </think>
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()
                else:
                    # If no closing tag, remove everything before first {
                    response = response[response.find("{"):]
            
            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                response = response[start:end+1]
            
            # Fix common issues
            response = re.sub(r',\s*}', '}', response)
            
            # If still empty or invalid, use fallback
            if not response or response == "{}":
                raise ValueError("Empty JSON response")
            
            result = json.loads(response)
            
            is_correct = bool(result.get('correct', False))
            confidence = float(result.get('confidence', 0.5))
            reasoning = str(result.get('reasoning', 'No reasoning provided'))
            
            logger.debug(f"LLM judgment: correct={is_correct}, confidence={confidence:.2f}")
            logger.debug(f"Reasoning: {reasoning}")
            
            return is_correct, confidence, reasoning
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Full response: {response}")
            
            # Fallback: Use emotional state
            if expected_state == "correct":
                return True, 0.8, "Fallback: student state indicates correctness"
            elif expected_state in ["confused", "frustrated", "disengaged"]:
                return False, 0.6, "Fallback: student state indicates confusion"
            else:
                return False, 0.5, "Fallback: neutral state"

class KnowledgeTracer:
    """
    Tracks student knowledge state using LLM-based correctness judgment.
    Maintains full conversation memory for context-aware evaluation.
    """
    
    def __init__(
        self,
        llm_judge: Optional[LLMCorrectnessJudge] = None,
        initial_mastery: float = 0.3
    ):
        """
        Initialize knowledge tracer.
        
        Args:
            llm_judge: LLM-based correctness judge (if None, uses simple heuristic)
            initial_mastery: Initial P(mastery)
        """
        self.llm_judge = llm_judge
        self.initial_mastery = initial_mastery
        self.knowledge_state = initial_mastery
        self.history = []
        self.full_conversation = []  # Store ALL turns with student + tutor messages
        
        logger.info(f"Knowledge Tracer initialized (initial mastery: {initial_mastery})")
    
    def update(
        self,
        problem: str,
        student_message: str,
        expected_state: str,
        tutor_action: str,
        tutor_response: Optional[str] = None
    ) -> float:
        """
        Update knowledge state based on student interaction.
        Maintains full conversation history.
        
        Args:
            problem: The math problem
            student_message: Student's message
            expected_state: Expected emotional state
            tutor_action: Action chosen by planner
            tutor_response: Optional tutor's response text
        
        Returns:
            Updated P(mastery)
        """
        # Add student message to conversation
        self.full_conversation.append({
            'role': 'student',
            'message': student_message
        })
        
        # Judge correctness using LLM with FULL conversation history
        if self.llm_judge:
            is_correct, confidence, reasoning = self.llm_judge.judge_correctness(
                problem=problem,
                student_message=student_message,
                expected_state=expected_state,
                full_conversation_history=self.full_conversation[:-1]  # All except current
            )
        else:
            # Simple fallback based on state
            is_correct = expected_state == "correct"
            confidence = 0.7
            reasoning = f"Fallback based on state: {expected_state}"
        
        # Update knowledge state using Bayesian-inspired update
        if is_correct:
            # Correct response → increase mastery
            # Confidence-weighted increment
            increment = 0.15 * confidence
            self.knowledge_state = min(1.0, self.knowledge_state + increment)
        else:
            # Incorrect/confused → slight decrease
            # Smaller decrement to avoid over-penalizing
            decrement = 0.05 * confidence
            self.knowledge_state = max(0.1, self.knowledge_state - decrement)
        
        # Record turn in history
        turn_record = {
            'turn': len(self.history) + 1,
            'message': student_message,
            'state': expected_state,
            'correct': is_correct,
            'confidence': confidence,
            'reasoning': reasoning,
            'mastery': self.knowledge_state,
            'action': tutor_action
        }
        
        self.history.append(turn_record)
        
        # Add tutor response to conversation if provided
        if tutor_response:
            self.full_conversation.append({
                'role': 'tutor',
                'message': tutor_response
            })
        
        logger.debug(
            f"Turn {len(self.history)}: correct={is_correct} "
            f"(conf={confidence:.2f}), mastery={self.knowledge_state:.3f}"
        )
        
        return self.knowledge_state
    
    def calculate_imv(self) -> float:
        """
        Calculate Inferred Mastery Velocity (IMV).
        
        IMV = (final_mastery - initial_mastery) / num_turns
        
        Higher IMV = Faster learning
        
        Returns:
            IMV score (mastery gained per turn)
        """
        if len(self.history) < 1:
            return 0.0
        
        final_mastery = self.knowledge_state
        num_turns = len(self.history)
        
        imv = (final_mastery - self.initial_mastery) / num_turns
        
        return imv
    
    def get_mastery_trajectory(self) -> List[float]:
        """
        Get mastery values over time.
        
        Returns:
            List of mastery probabilities at each turn
        """
        return [h['mastery'] for h in self.history]
    
    def get_final_mastery(self) -> float:
        """Get final mastery level."""
        return self.knowledge_state
    
    def get_detailed_history(self) -> List[Dict]:
        """Get full history with all details."""
        return self.history
    
    def reset(self):
        """Reset knowledge state for new scenario."""
        self.knowledge_state = self.initial_mastery
        self.history = []
        self.full_conversation = []


def calculate_avg_imv(scenario_results: List[Dict]) -> float:
    """
    Calculate average IMV across all scenarios.
    
    Args:
        scenario_results: List of scenario results with IMV values
    
    Returns:
        Average IMV
    """
    imvs = [result.get('imv', 0.0) for result in scenario_results if 'imv' in result]
    return float(np.mean(imvs)) if imvs else 0.0


def calculate_avg_final_mastery(scenario_results: List[Dict]) -> float:
    """
    Calculate average final mastery across all scenarios.
    
    Args:
        scenario_results: List of scenario results
    
    Returns:
        Average final mastery
    """
    final_masteries = [
        result.get('final_mastery', 0.0) 
        for result in scenario_results 
        if 'final_mastery' in result
    ]
    return float(np.mean(final_masteries)) if final_masteries else 0.0


if __name__ == "__main__":
    """Test knowledge tracer with Qwen3-8B."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 70)
    print("INITIALIZING KNOWLEDGE TRACER WITH QWEN3-8B")
    print("=" * 70)
    
    # Initialize LLM judge with Qwen3-8B
    llm_judge = LLMCorrectnessJudge(
        model_path="Qwen/Qwen3-8B",
        use_quantization=True
    )
    
    # Initialize tracer
    tracer = KnowledgeTracer(llm_judge=llm_judge)
    
    # Simulate a 10-turn conversation
    problem = "Solve for x: 3x + 7 = 22"
    
    turns = [
        ("I don't know where to start with this problem", "confused"),
        ("Do I need to subtract something?", "confused"),
        ("Should I subtract 7 from both sides?", "confused"),
        ("Ok so I get 3x = 15", "neutral"),
        ("Wait is that right? 22 - 7 = 15?", "neutral"),
        ("Yes! So 3x = 15", "neutral"),
        ("Now what do I do with the 3?", "confused"),
        ("Oh I divide both sides by 3!", "neutral"),
        ("So x = 5", "correct"),
        ("Let me check: 3(5) + 7 = 15 + 7 = 22. Yes!", "correct"),
    ]
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE TRACING SIMULATION (10 TURNS)")
    print("=" * 70)
    print(f"Problem: {problem}\n")
    
    for i, (message, state) in enumerate(turns):
        mastery = tracer.update(
            problem=problem,
            student_message=message,
            expected_state=state,
            tutor_action="provide_hint",
            tutor_response=f"[Tutor provides guidance for turn {i+1}]"
        )
        
        print(f"Turn {i+1}: '{message}'")
        print(f"  State: {state:12s} | Mastery: {mastery:.3f}")
    
    imv = tracer.calculate_imv()
    trajectory = tracer.get_mastery_trajectory()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"IMV (Inferred Mastery Velocity): {imv:.4f} mastery/turn")
    print(f"Initial Mastery: {trajectory[0]:.3f}")
    print(f"Final Mastery:   {trajectory[-1]:.3f}")
    print(f"Total Gain:      {trajectory[-1] - trajectory[0]:.3f}")
    print(f"\nMastery Trajectory:")
    print(f"  {' → '.join([f'{m:.3f}' for m in trajectory])}")
    print("=" * 70)