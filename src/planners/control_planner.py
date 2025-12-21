"""
Control Planner: Standard Tree-of-Thought Baseline with Full Metrics
Uses BASE Qwen3-8B (non-finetuned) with single-prompt planning.
Integrates trained models for comprehensive metric calculation.
"""

import json
import time
import torch
import numpy as np
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


class ControlPlanner:
    """
    Standard Tree-of-Thought planner using base Qwen3-8B.
    Baseline for comparison with Experimental Planner (DPO-finetuned).
    Includes full metric calculation.
    """
    
    def __init__(
        self,
        use_quantization: bool = True,
        student_state_classifier_path: Optional[str] = None,
        alignment_scorer_path: Optional[str] = None
    ):
        """
        Initialize Control Planner with base Qwen3-8B and metric models.
        
        Args:
            use_quantization: Use 4-bit NF4 quantization (for 8GB GPU)
            student_state_classifier_path: Path to student state classifier
            alignment_scorer_path: Path to alignment scorer model
        """
        print("=" * 80)
        print("INITIALIZING CONTROL PLANNER WITH METRICS")
        print("Model: Base Qwen3-8B (non-finetuned)")
        print("=" * 80)
        
        MODEL_NAME = "Qwen/Qwen3-8B"
        
        # Load base model
        print(f"\n1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"\n2. Loading base Qwen3-8B model...")
        
        if use_quantization:
            print("   Using NF4 4-bit quantization")
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
        
        # Load metric models
        self._load_metric_models(student_state_classifier_path, alignment_scorer_path)
        
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
        
        # Logging
        self.turn_logs = []
        self.student_state_history = []  # For NARR calculation
        
        print("\n✅ Control Planner initialized!")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   VRAM allocated: {allocated:.2f} GB")
        print("=" * 80)
    
    def _load_metric_models(
        self,
        student_state_classifier_path: Optional[str],
        alignment_scorer_path: Optional[str]
    ):
        """Load models needed for metric calculation."""
        
        # Student State Classifier (for NARR)
        self.student_state_classifier = None
        if student_state_classifier_path:
            print(f"\n3. Loading Student State Classifier...")
            try:
                # TODO: Load your actual student_state_classifier
                # For now, placeholder
                print(f"   ⚠️  Placeholder - implement k-NN classifier loading")
                # self.student_state_classifier = load_knn_classifier(student_state_classifier_path)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not load student state classifier: {e}")
        
        # Alignment Scorer (for Pedagogical Alignment Score)
        self.alignment_scorer = None
        self.embedding_model = None
        if alignment_scorer_path:
            print(f"\n4. Loading Alignment Scorer...")
            try:
                with open(alignment_scorer_path, 'rb') as f:
                    self.alignment_scorer = pickle.load(f)
                
                # Load embedding model (same as used in training)
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"   ✅ Alignment scorer loaded")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not load alignment scorer: {e}")
    
    def plan_action(
        self,
        conversation_history: List[Dict[str, str]],
        current_problem: str,
        turn_number: int,
        max_new_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Main planning function - single LLM call does everything.
        
        Args:
            conversation_history: List of {"role": "student/tutor", "message": "..."}
            current_problem: The math problem being worked on
            turn_number: Current turn number
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Complete result dict with metrics
        """
        start_time = time.time()
        
        # Get latest student message
        latest_student_msg = self._get_latest_student_message(conversation_history)
        
        # METRIC: Assess student state (for NARR)
        student_state = self._assess_student_state(latest_student_msg)
        
        # Format conversation history
        history_str = self._format_history(conversation_history)
        
        # Build the single comprehensive prompt
        prompt = self._build_control_prompt(
            history_str=history_str,
            current_problem=current_problem,
            latest_student_msg=latest_student_msg
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_tokens = inputs.input_ids.shape[1]
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        print(f"\n🤖 Control Planner generating decision (Turn {turn_number})...")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
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
        
        # METRIC: Calculate Pedagogical Alignment Score
        alignment_score = self._calculate_alignment_score(parsed["reasoning"])
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"⏱️  Generation took {latency_ms/1000:.1f} seconds")
        
        # Build result with all metrics
        result = {
            # Turn info
            "turn_number": turn_number,
            "selected_action": parsed["selected_action"],
            "reasoning": parsed["reasoning"],
            "effectiveness_score": parsed["effectiveness_score"],
            "all_candidates": parsed["all_candidates"],
            "student_state_analysis": parsed.get("student_state_analysis", ""),
            "scored_candidates": parsed.get("scored_candidates", []),
            
            # Student state (for NARR)
            "student_state": student_state,
            
            # EFFICIENCY METRICS
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            
            # EFFECTIVENESS METRIC
            "pedagogical_alignment_score": alignment_score,
            
            # Metadata
            "timestamp": time.time(),
            "planner_type": "control"
        }
        
        # Log
        self.turn_logs.append(result)
        self.student_state_history.append(student_state)
        
        return result
    
    def _assess_student_state(self, student_message: str) -> Dict[str, float]:
        """
        Assess student's emotional/cognitive state.
        
        Args:
            student_message: The student's latest message
        
        Returns:
            Dict with state probabilities (e.g., {'is_confused': 0.8, ...})
        """
        if self.student_state_classifier is None:
            # Fallback: simple keyword-based heuristic
            msg_lower = student_message.lower()
            
            is_frustrated = any(word in msg_lower for word in ['frustrated', 'stuck', 'can\'t', 'don\'t understand'])
            is_confused = any(word in msg_lower for word in ['confused', 'what', 'how', 'why', '?'])
            is_correct = any(word in msg_lower for word in ['got it', 'understand', 'makes sense', 'i see'])
            
            return {
                'is_frustrated': 0.8 if is_frustrated else 0.2,
                'is_confused': 0.7 if is_confused else 0.3,
                'is_correct': 0.7 if is_correct else 0.3,
                'is_disengaged': 0.2,
                'is_neutral': 0.5
            }
        else:
            # Use actual classifier
            # TODO: Implement with your k-NN classifier
            return self.student_state_classifier.predict(student_message)
    
    def _calculate_alignment_score(self, reasoning_text: str) -> float:
        """
        Calculate pedagogical alignment score for the reasoning.
        
        Args:
            reasoning_text: The generated reasoning/deliberation
        
        Returns:
            Alignment score (0-1)
        """
        if self.alignment_scorer is None or self.embedding_model is None:
            return 0.0  # No score available
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([reasoning_text])
            
            # Predict score
            score = self.alignment_scorer.predict(embedding)[0]
            
            # Clip to valid range
            return float(np.clip(score, 0.0, 1.0))
        
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate alignment score: {e}")
            return 0.0
    
    def _build_control_prompt(
        self,
        history_str: str,
        current_problem: str,
        latest_student_msg: str
    ) -> str:
        """Build the single comprehensive prompt for the Control Planner."""
        
        prompt = f"""You are a pedagogical AI tutor for mathematics (Algebra, Grades 5-12). You are using a Tree-of-Thought planning approach to decide your next teaching action.

**PEDAGOGICAL CONSTITUTION:**
1. Foster Constructivism & Scaffolding - Guide, don't just tell
2. Manage Cognitive Load - Don't overwhelm the student
3. Maintain Desirable Difficulty - Keep student in the zone
4. Promote Metacognition - Encourage reflection on thinking
5. Foster Positive Affect & Validation - Keep student willing to learn

**PEDAGOGICAL ACTIONS AVAILABLE:**
{', '.join(self.pedagogical_actions)}

**CURRENT PROBLEM:**
{current_problem}

**CONVERSATION HISTORY:**
{history_str}

**LATEST STUDENT MESSAGE:**
{latest_student_msg}

**YOUR TASK:**
Follow these steps internally (show your work):

**STEP A: ANALYZE STUDENT STATE**
Analyze the student's current state. Consider:
- Are they confused, frustrated, correct, stuck, or engaged?
- What misconception or knowledge gap might they have?
- What is their emotional state?

**STEP B: GENERATE CANDIDATE ACTIONS**
Generate 3-5 candidate pedagogical actions from the list above. For each candidate, briefly explain why it might be appropriate.

**STEP C: EVALUATE CANDIDATES**
For each candidate action, assign a "Pedagogical Effectiveness Score" from 1-10 based on:
- How well it addresses the student's current state
- How likely it is to move learning forward
- Alignment with pedagogical best practices

**STEP D: SELECT BEST ACTION**
Choose the action with the highest effectiveness score as your final decision.

**OUTPUT FORMAT (JSON):**
Return your response as a valid JSON object with this exact structure:

{{
  "step_a_analysis": "Your analysis of student state...",
  "step_b_candidates": [
    {{
      "action": "ask_socratic_question",
      "rationale": "Why this might work..."
    }},
    {{
      "action": "provide_hint",
      "rationale": "Why this might work..."
    }},
    {{
      "action": "simplify_problem",
      "rationale": "Why this might work..."
    }}
  ],
  "step_c_scores": [
    {{
      "action": "ask_socratic_question",
      "effectiveness_score": 8,
      "justification": "Why this score..."
    }},
    {{
      "action": "provide_hint",
      "effectiveness_score": 6,
      "justification": "Why this score..."
    }},
    {{
      "action": "simplify_problem",
      "effectiveness_score": 7,
      "justification": "Why this score..."
    }}
  ],
  "step_d_selection": {{
    "selected_action": "ask_socratic_question",
    "reasoning": "Final reasoning for why this is best...",
    "effectiveness_score": 8
  }}
}}

**IMPORTANT:**
- Return ONLY the JSON object, no markdown formatting, no extra text
- Ensure valid JSON syntax
- Include all 4 steps (A, B, C, D) in your response
- Generate at least 3 candidate actions

RESPONSE:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response."""
        
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            
            return {
                "selected_action": parsed["step_d_selection"]["selected_action"],
                "reasoning": parsed["step_d_selection"]["reasoning"],
                "effectiveness_score": parsed["step_d_selection"]["effectiveness_score"],
                "all_candidates": parsed["step_b_candidates"],
                "student_state_analysis": parsed.get("step_a_analysis", ""),
                "scored_candidates": parsed.get("step_c_scores", [])
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:500]}...")
            return self._fallback_response()
    
    def _fallback_response(self) -> Dict[str, Any]:
        """Return a safe fallback if parsing fails."""
        return {
            "selected_action": "provide_hint",
            "reasoning": "Fallback action due to parsing error",
            "effectiveness_score": 5.0,
            "all_candidates": [],
            "student_state_analysis": "Error in analysis",
            "scored_candidates": []
        }
    
    def _format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history as readable text."""
        if not conversation_history:
            return "No previous conversation."
        
        lines = []
        for turn in conversation_history:
            role = turn["role"].capitalize()
            message = turn["message"]
            lines.append(f"{role}: {message}")
        return "\n".join(lines)
    
    def _get_latest_student_message(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract the most recent student message."""
        for turn in reversed(conversation_history):
            if turn["role"] == "student":
                return turn["message"]
        return "No student message yet."
    
    def calculate_narr(self) -> float:
        """
        Calculate Negative Affect Reduction Rate (NARR).
        
        Returns:
            NARR score (0-1), percentage of negative states successfully resolved
        """
        if len(self.student_state_history) < 2:
            return 0.0
        
        triggers = []
        resolutions = []
        
        # Identify trigger events (high frustration/confusion)
        for i in range(len(self.student_state_history) - 1):
            state = self.student_state_history[i]
            
            # Trigger: is_frustrated > 0.7
            if state.get('is_frustrated', 0) > 0.7:
                triggers.append(i)
                
                # Check next state (after tutor intervention)
                if i + 1 < len(self.student_state_history):
                    next_state = self.student_state_history[i + 1]
                    
                    # Resolution: frustration decreased by > 30%
                    reduction = state['is_frustrated'] - next_state.get('is_frustrated', 0)
                    if reduction > 0.3:
                        resolutions.append(i)
        
        if len(triggers) == 0:
            return 1.0  # No negative states encountered
        
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
        """
        Get summary of all metrics across all turns.
        
        Returns:
            Dictionary with all metric summaries
        """
        if not self.turn_logs:
            return {}
        
        # EFFICIENCY METRICS
        latencies = [log["latency_ms"] for log in self.turn_logs]
        total_tokens_list = [log["total_tokens"] for log in self.turn_logs]
        
        # EFFECTIVENESS METRICS
        alignment_scores = [
            log["pedagogical_alignment_score"] 
            for log in self.turn_logs 
            if log["pedagogical_alignment_score"] > 0
        ]
        
        # ROBUSTNESS METRICS
        narr = self.calculate_narr()
        
        return {
            # Basic info
            "total_turns": len(self.turn_logs),
            "planner_type": "control",
            
            # EFFICIENCY METRICS
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "avg_tokens_per_turn": np.mean(total_tokens_list),
            "total_tokens_used": sum(total_tokens_list),
            
            # EFFECTIVENESS METRICS
            "avg_pedagogical_alignment_score": np.mean(alignment_scores) if alignment_scores else 0.0,
            "std_pedagogical_alignment_score": np.std(alignment_scores) if alignment_scores else 0.0,
            
            # ROBUSTNESS METRICS
            "negative_affect_reduction_rate": narr
        }


# Example usage and testing
if __name__ == "__main__":
    """Test the Control Planner with metrics."""
    
    print("\n🚀 Initializing Control Planner with Metrics...\n")
    
    # Initialize planner with alignment scorer
    planner = ControlPlanner(
        use_quantization=True,
        alignment_scorer_path="models/alignment_scorer/alignment_scorer.pkl"
    )
    
    # Test Scenario 1: Confused student
    print("\n" + "=" * 80)
    print("TEST SCENARIO 1: Confused Student")
    print("=" * 80)
    
    conversation_history = [
        {"role": "tutor", "message": "Let's solve: 3x + 7 = 22. What's your first step?"},
        {"role": "student", "message": "I subtracted 7 from both sides and got 3x = 15, but now I'm stuck. What do I do with the 3?"}
    ]
    
    result = planner.plan_action(
        conversation_history=conversation_history,
        current_problem="Solve for x: 3x + 7 = 22",
        turn_number=1
    )
    
    print(f"\n📊 PLANNER DECISION:")
    print(f"   Selected Action: {result['selected_action']}")
    print(f"   Effectiveness Score: {result['effectiveness_score']}/10")
    print(f"   Reasoning: {result['reasoning'][:200]}...")
    
    print(f"\n📈 METRICS:")
    print(f"   Latency: {result['latency_ms']:.0f} ms")
    print(f"   Tokens: {result['total_tokens']}")
    print(f"   Pedagogical Alignment Score: {result['pedagogical_alignment_score']:.3f}")
    print(f"   Student State: frustrated={result['student_state']['is_frustrated']:.2f}, confused={result['student_state']['is_confused']:.2f}")
    
    # Test Scenario 2: Frustrated student
    print("\n" + "=" * 80)
    print("TEST SCENARIO 2: Frustrated Student")
    print("=" * 80)
    
    conversation_history2 = [
        {"role": "tutor", "message": "Factor: x² + 5x + 6"},
        {"role": "student", "message": "I keep trying to factor this but I can't find two numbers that work! This is so frustrating!"}
    ]
    
    result2 = planner.plan_action(
        conversation_history=conversation_history2,
        current_problem="Factor: x² + 5x + 6",
        turn_number=2
    )
    
    print(f"\n📊 PLANNER DECISION:")
    print(f"   Selected Action: {result2['selected_action']}")
    print(f"   Alignment Score: {result2['pedagogical_alignment_score']:.3f}")
    
    # Save logs
    planner.save_logs("logs/control_planner/test_run.json")
    
    # Print comprehensive metrics
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE METRICS SUMMARY:")
    print("=" * 80)
    metrics = planner.get_metrics_summary()
    
    print(f"\n🎯 EFFICIENCY METRICS:")
    print(f"   Total Turns: {metrics['total_turns']}")
    print(f"   Avg Latency: {metrics['avg_latency_ms']:.0f} ms (±{metrics['std_latency_ms']:.0f})")
    print(f"   Avg Tokens/Turn: {metrics['avg_tokens_per_turn']:.0f}")
    print(f"   Total Tokens Used: {metrics['total_tokens_used']}")
    
    print(f"\n✅ EFFECTIVENESS METRICS:")
    print(f"   Avg Pedagogical Alignment Score: {metrics['avg_pedagogical_alignment_score']:.3f} (±{metrics['std_pedagogical_alignment_score']:.3f})")
    
    print(f"\n💪 ROBUSTNESS METRICS:")
    print(f"   Negative Affect Reduction Rate (NARR): {metrics['negative_affect_reduction_rate']:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ CONTROL PLANNER TEST COMPLETE")
    print("=" * 80)