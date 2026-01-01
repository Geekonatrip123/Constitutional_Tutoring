"""
Control Planner: Vanilla Tree-of-Thought Baseline with Knowledge Tracing
Uses existing knowledge_tracer.py module.
"""

import json
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import existing knowledge tracer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.evaluation.knowledge_tracer import KnowledgeTracer, LLMCorrectnessJudge


class ControlPlanner:
    """
    Vanilla Tree-of-Thought planner using SINGLE base Qwen3-8B.
    Uses existing KnowledgeTracer for learning metrics.
    """

    def __init__(
        self,
        use_quantization: bool = True,
        enable_knowledge_tracing: bool = True
    ):

        MODEL_NAME = "Qwen/Qwen3-8B"

        print(f"\n1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        self.model.eval()

        # Pedagogical actions
        self.pedagogical_actions = [
            "ask_socratic_question",
            "provide_hint",
            "show_worked_example",
            "reframe_problem",
            "ask_for_explanation",
            "define_key_term",
            "give_positive_feedback",
            "increase_difficulty",
            "simplify_problem",
            "give_direct_answer"
        ]

        # Knowledge Tracing using existing module
        self.enable_knowledge_tracing = enable_knowledge_tracing
        self.knowledge_tracer = None
        
        if enable_knowledge_tracing:
            print(f"\n3. Initializing Knowledge Tracer...")
            llm_judge = LLMCorrectnessJudge(
                model=self.model,
                tokenizer=self.tokenizer,
                use_shared_model=True  # Share model with planner
            )
            self.knowledge_tracer = KnowledgeTracer(
                llm_judge=llm_judge,
                initial_mastery=0.35
            )

            import types
            self.knowledge_tracer.update = types.MethodType(
                self.knowledge_tracer
            )
            print(" Knowledge Tracer initialized")

        # Logging
        self.turn_logs = []
        self.student_state_history = []
        self.current_problem = None

        print("\n✅ Vanilla Control Planner initialized!")
        print("   SINGLE base Qwen2.5-8B model for everything")
        print("   NO alignment scorer (true baseline)")
        print("   NO specialized models")
        if enable_knowledge_tracing:
            print("   WITH knowledge tracing (using shared model)")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   VRAM allocated: {allocated:.2f} GB")
        print("=" * 80)

    def plan_action(
        self,
        conversation_history: List[Dict[str, str]],
        current_problem: str,
        turn_number: int,
        max_new_tokens: int = 2048,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Main planning function using existing KnowledgeTracer."""
        start_time = time.time()

        # Store current problem
        if self.current_problem is None:
            self.current_problem = current_problem

        # Get latest student message
        latest_student_msg = self._get_latest_student_message(conversation_history)

        # Simple heuristic student state (NO classifier)
        student_state = self._assess_student_state_heuristic(latest_student_msg)

        # Build the simple Tree-of-Thought prompt
        prompt = self._build_control_prompt(
            current_problem=current_problem,
            latest_student_msg=latest_student_msg
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_tokens = inputs.input_ids.shape[1]

        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print(f"\n Control Planner generating decision (Turn {turn_number})...")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_output[len(prompt):].strip()

        # Count output tokens
        output_tokens = len(self.tokenizer.encode(response_text))
        total_tokens = input_tokens + output_tokens

        # Parse the structured response
        parsed = self._parse_response(response_text)

        kt_result = {}
        if self.knowledge_tracer is not None:
            # Determine expected state from heuristic
            expected_state = self._determine_expected_state(student_state)
            
            # Update knowledge tracer
            new_mastery = self.knowledge_tracer.update(
                problem=current_problem,
                student_message=latest_student_msg,
                expected_state=expected_state,
                tutor_action=parsed["selected_action"],
                tutor_response=f"[Tutor performs: {parsed['selected_action']}]"
            )
            
            # Get latest turn info
            if self.knowledge_tracer.history:
                latest_turn = self.knowledge_tracer.history[-1]
                kt_result = {
                    'is_correct': latest_turn['correct'],
                    'confidence': latest_turn['confidence'],
                    'mastery_probability': new_mastery,
                    'reasoning': latest_turn['reasoning']
                }

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000


        # Build result
        result = {
            "turn_number": turn_number,
            "selected_action": parsed["selected_action"],
            "reasoning": parsed["reasoning"],
            "effectiveness_score": parsed["effectiveness_score"],
            "all_candidates": parsed["all_candidates"],
            "student_state": student_state,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "pedagogical_alignment_score": 0.0,  # NOT CALCULATED
            "knowledge_tracing": kt_result,
            "timestamp": time.time(),
            "planner_type": "control_vanilla_tot"
        }

        # Log
        self.turn_logs.append(result)
        self.student_state_history.append(student_state)

        return result

    def _determine_expected_state(self, student_state: Dict[str, float]) -> str:
        """Determine expected emotional state from heuristic scores."""
        if student_state.get('is_correct', 0) > 0.6:
            return 'correct'
        elif student_state.get('is_frustrated', 0) > 0.6:
            return 'frustrated'
        elif student_state.get('is_confused', 0) > 0.6:
            return 'confused'
        elif student_state.get('is_disengaged', 0) > 0.6:
            return 'disengaged'
        else:
            return 'neutral'

    def _assess_student_state_heuristic(self, student_message: str) -> Dict[str, float]:
        """Simple heuristic-based student state assessment."""
        msg_lower = student_message.lower()

        is_frustrated = any(word in msg_lower for word in [
            'frustrated', 'stuck', "can't", "don't understand", 
            "don't get", "confused", "help"
        ])
        
        is_confused = any(word in msg_lower for word in [
            'confused', 'what', 'how', 'why', '?', 'not sure', 'unclear'
        ])
        
        is_correct = any(word in msg_lower for word in [
            'got it', 'understand', 'makes sense', 'i see', 
            'oh', 'right', 'yes', 'correct'
        ])
        
        is_disengaged = any(word in msg_lower for word in [
            'whatever', 'idk', "don't care", 'boring', 'skip'
        ])

        return {
            'is_frustrated': 0.8 if is_frustrated else 0.2,
            'is_confused': 0.7 if is_confused else 0.3,
            'is_correct': 0.7 if is_correct else 0.3,
            'is_disengaged': 0.7 if is_disengaged else 0.2,
            'is_neutral': 0.5
        }

    def _build_control_prompt(
        self,
        current_problem: str,
        latest_student_msg: str
    ) -> str:
        """Build SIMPLE Tree-of-Thought prompt - vanilla baseline."""

        prompt = f"""You are a math tutor helping a student solve a problem.

PROBLEM: {current_problem}

STUDENT'S LAST MESSAGE: {latest_student_msg}

AVAILABLE TEACHING ACTIONS:
{', '.join(self.pedagogical_actions)}

Think through this step-by-step using Tree-of-Thought reasoning:

Step 1: Analyze what the student needs right now.

Step 2: Generate 3 possible teaching actions you could take.

Step 3: For each action, assign an effectiveness score (1-10),it can be in decimals too based on how well it would help the student learn.
IMPORTANT: Use the FULL range of scores. Don't just give everything a 7 or 8. Be critical and realistic:
- Scores 1-3: Poor fit for this situation
- Scores 4-6: Adequate but not ideal
- Scores 7-8: Good fit for the situation
- Scores 9-10: Excellent, highly appropriate action

Step 4: Select the action with the HIGHEST score.

Provide your response in this JSON format:
{{
  "step1_analysis": "brief analysis of student's needs",
  "step2_options": [
    {{"action": "action_name_1", "reasoning": "why this could work"}},
    {{"action": "action_name_2", "reasoning": "why this could work"}},
    {{"action": "action_name_3", "reasoning": "why this could work"}}
  ],
  "step3_scores": [
    {{"action": "action_name_1", "score": 5, "justification": "brief reason"}},
    {{"action": "action_name_2", "score": 7, "justification": "brief reason"}},
    {{"action": "action_name_3", "score": 4, "justification": "brief reason"}}
  ],
  "step4_final": {{
    "selected_action": "action_name_2",
    "reasoning": "final explanation for this choice",
    "effectiveness_score": 7
  }}
}}

JSON:"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response."""
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            import json.decoder
            decoder = json.decoder.JSONDecoder()

            # Find JSON object
            start_idx = response_text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found")

            parsed, end_idx = decoder.raw_decode(response_text, start_idx)

            # Extract key fields
            return {
                "selected_action": parsed["step4_final"]["selected_action"],
                "reasoning": parsed["step4_final"]["reasoning"],
                "effectiveness_score": parsed["step4_final"]["effectiveness_score"],
                "all_candidates": parsed.get("step2_options", []),
                "student_state_analysis": parsed.get("step1_analysis", "")
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f" Warning: Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:500]}...")
            return self._fallback_response()

    def _fallback_response(self) -> Dict[str, Any]:
        """Return a safe fallback if parsing fails."""
        return {
            "selected_action": "provide_hint",
            "reasoning": "Fallback action due to parsing error",
            "effectiveness_score": 5.0,
            "all_candidates": [],
            "student_state_analysis": "Unable to parse analysis"
        }

    def _get_latest_student_message(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract the most recent student message."""
        for turn in reversed(conversation_history):
            if turn["role"] == "student":
                return turn["message"]
        return "No student message yet."

    def reset_for_new_scenario(self, problem: str):
        """Reset planner state for a new tutoring scenario."""
        self.turn_logs = []
        self.student_state_history = []
        self.current_problem = problem
        
        # Reset knowledge tracer
        if self.knowledge_tracer is not None:
            self.knowledge_tracer.reset()
        
        print(f"\n🔄 Control Planner reset for new problem: {problem[:50]}...")

    def calculate_narr(self) -> float:
        """Calculate Negative Affect Reduction Rate (NARR)."""
        if len(self.student_state_history) < 2:
            return 0.0

        triggers = []
        resolutions = []

        for i in range(len(self.student_state_history) - 1):
            state = self.student_state_history[i]

            negative_affect = max(
                state.get('is_frustrated', 0),
                state.get('is_confused', 0)
            )

            if negative_affect > 0.6:
                triggers.append(i)

                if i + 1 < len(self.student_state_history):
                    next_state = self.student_state_history[i + 1]
                    next_negative_affect = max(
                        next_state.get('is_frustrated', 0),
                        next_state.get('is_confused', 0)
                    )
                    
                    if negative_affect - next_negative_affect > 0.3:
                        resolutions.append(i)

        if len(triggers) == 0:
            return 1.0

        narr = len(resolutions) / len(triggers)
        return narr

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return all turn logs for analysis."""
        return self.turn_logs

    def save_logs(self, output_path: str):
        """Save logs to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.turn_logs, f, indent=2)
        print(f"✅ Logs saved to: {output_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics using KnowledgeTracer."""
        if not self.turn_logs:
            return {}

        latencies = [log["latency_ms"] for log in self.turn_logs]
        total_tokens_list = [log["total_tokens"] for log in self.turn_logs]

        narr = self.calculate_narr()

        # Get IMV and mastery from KnowledgeTracer
        imv = 0.0
        final_mastery = 0.35
        mastery_trajectory = []

        if self.knowledge_tracer is not None:
            imv = self.knowledge_tracer.calculate_imv()
            final_mastery = self.knowledge_tracer.get_final_mastery()
            mastery_trajectory = self.knowledge_tracer.get_mastery_trajectory()

        return {
            "total_turns": len(self.turn_logs),
            "planner_type": "control_vanilla_tot",
            
            # Efficiency metrics
            "avg_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "avg_tokens_per_turn": float(np.mean(total_tokens_list)),
            "total_tokens_used": int(sum(total_tokens_list)),
            
            # Effectiveness metrics
            "avg_pedagogical_alignment_score": 0.0,  # N/A
            "std_pedagogical_alignment_score": 0.0,  # N/A
            "imv": float(imv),  # From KnowledgeTracer
            "final_mastery": float(final_mastery),  # From KnowledgeTracer
            "mastery_trajectory": mastery_trajectory,
            
            # Transparency metrics
            "deliberation_action_congruence": 0.0,  # N/A
            "principle_coverage_frequency": {},  # N/A
            
            # Robustness metrics
            "negative_affect_reduction_rate": float(narr)
        }