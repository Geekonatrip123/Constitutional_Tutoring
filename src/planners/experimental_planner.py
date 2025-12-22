"""
Experimental Planner: Deliberative Hybrid Architecture
4-stage pipeline using all trained models for transparent, principled decision-making.
"""

import json
import time
import torch
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer


class ExperimentalPlanner:
    """
    Deliberative Hybrid Planner using 4-stage pipeline:
    Stage 1: Student State Classification (student_state_classifier)
    Stage 2: Deliberation Generation (DPO-finetuned Qwen3-8B)
    Stage 3: Alignment Scoring (alignment_scorer)
    Stage 4: Final Selection (synthesizing LLM)
    """

    def __init__(
        self,
        student_state_classifier_path: Optional[str] = None,
        deliberation_generator_path: str = None,
        alignment_scorer_path: str = None,
        use_quantization: bool = True
    ):
        """
        Initialize Experimental Planner with all trained models.
        
        Args:
            student_state_classifier_path: Path to SEC best_model.pt
            deliberation_generator_path: Path to DPO-finetuned model (LoRA adapters)
            alignment_scorer_path: Path to alignment scorer
            use_quantization: Use 4-bit NF4 quantization
        """
        print("=" * 80)
        print("INITIALIZING EXPERIMENTAL PLANNER")
        print("4-Stage Deliberative Hybrid Architecture")
        print("=" * 80)

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

        # Pedagogical Constitution
        self.pedagogical_principles = {
            1: "Foster Constructivism & Scaffolding",
            2: "Manage Cognitive Load",
            3: "Maintain Desirable Difficulty",
            4: "Promote Metacognition",
            5: "Foster Positive Affect & Validation",
            6: "Maintain Factual Integrity"
        }

        # Stage 1: Student State Classifier
        self._load_student_state_classifier(student_state_classifier_path)

        # Stage 2: Deliberation Generator (DPO-finetuned)
        self._load_deliberation_generator(deliberation_generator_path, use_quantization)

        # Stage 3: Alignment Scorer
        self._load_alignment_scorer(alignment_scorer_path)

        # Logging
        self.turn_logs = []
        self.student_state_history = []

        print("\n✅ Experimental Planner initialized!")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   VRAM allocated: {allocated:.2f} GB")
        print("=" * 80)

    def _load_student_state_classifier(self, classifier_path: Optional[str]):
        """Load Stage 1: Student State Classifier."""
        print(f"\n[STAGE 1] Loading Student State Classifier...")

        self.student_state_classifier = None

        if classifier_path:
            try:
                import sys
                sys.path.append(str(Path(__file__).parent.parent.parent))
                
                # Import your existing classifier
                from src.models.student_state_classifier import load_student_state_classifier
                from src.utils.llm_utils import GeminiAPI
                from src.utils.config import load_config

                # Load config for Gemini API
                print(f"   Loading configuration...")
                config = load_config()
                gemini = GeminiAPI(config.api_keys.gemini.to_dict())

                # Load complete classifier
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                print(f"   Loading SEC model and k-NN classifier...")

                self.student_state_classifier = load_student_state_classifier(
                    sec_checkpoint_path=classifier_path,
                    tokenizer_dir=str(Path(classifier_path).parent / "tokenizer"),
                    train_embeddings_path="src/data/processed/train_embeddings.npy",
                    train_labels_path="src/data/processed/train_labels.npy",
                    llm_api=gemini,
                    device=device,
                    k=5
                )

                print(f"   ✅ Student state classifier loaded")

            except Exception as e:
                print(f"   ⚠️  Warning: Could not load classifier: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Using heuristic fallback")
        else:
            print(f"   ⚠️  Using heuristic fallback (no classifier path provided)")

    def _load_deliberation_generator(self, generator_path: str, use_quantization: bool):
        """Load Stage 2: DPO-finetuned Deliberation Generator."""
        print(f"\n[STAGE 2] Loading Deliberation Generator (DPO-finetuned)...")

        if not generator_path:
            raise ValueError("deliberation_generator_path is required!")

        from peft import PeftConfig

        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(generator_path)
        BASE_MODEL = peft_config.base_model_name_or_path
        print(f"   Base model: {BASE_MODEL}")

        # Load tokenizer
        self.delib_tokenizer = AutoTokenizer.from_pretrained(
            generator_path,
            trust_remote_code=True
        )
        if self.delib_tokenizer.pad_token is None:
            self.delib_tokenizer.pad_token = self.delib_tokenizer.eos_token

        # Load base model
        if use_quantization:
            print(f"   Using NF4 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        # Load LoRA adapters (DPO-finetuned)
        print(f"   Loading DPO LoRA adapters...")
        self.delib_model = PeftModel.from_pretrained(
            base_model,
            generator_path,
            is_trainable=False
        )
        self.delib_model.eval()
        print(f"   ✅ DPO-finetuned model loaded")

    def _load_alignment_scorer(self, scorer_path: str):
        """Load Stage 3: Alignment Scorer."""
        print(f"\n[STAGE 3] Loading Alignment Scorer...")

        if not scorer_path:
            raise ValueError("alignment_scorer_path is required!")

        try:
            with open(scorer_path, 'rb') as f:
                self.alignment_scorer = pickle.load(f)

            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"   ✅ Alignment scorer loaded")
        except Exception as e:
            raise ValueError(f"Could not load alignment scorer: {e}")

    def plan_action(
        self,
        conversation_history: List[Dict[str, str]],
        current_problem: str,
        turn_number: int,
        num_candidates: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Main planning function - 4-stage deliberative pipeline.

        Args:
            conversation_history: List of {"role": "student/tutor", "message": "..."}
            current_problem: The math problem being worked on
            turn_number: Current turn number
            num_candidates: Number of candidate actions to generate
            max_new_tokens: Max tokens for generation
            temperature: Sampling temperature

        Returns:
            Complete result dict with all stages and metrics
        """
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"EXPERIMENTAL PLANNER - TURN {turn_number}")
        print(f"{'='*80}")

        # Get latest student message
        latest_student_msg = self._get_latest_student_message(conversation_history)

        # ===== STAGE 1: STRUCTURED STATE ASSESSMENT =====
        stage1_start = time.time()
        student_state = self._stage1_assess_state(latest_student_msg)
        stage1_time = (time.time() - stage1_start) * 1000
        print(f"\n[STAGE 1] ✅ State assessed ({stage1_time:.0f}ms)")
        print(f"   {student_state}")

        # ===== STAGE 2: PRINCIPLED DELIBERATION & CANDIDATE GENERATION =====
        stage2_start = time.time()
        candidates = self._stage2_generate_deliberations(
            student_state=student_state,
            conversation_history=conversation_history,
            current_problem=current_problem,
            latest_student_msg=latest_student_msg,
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        stage2_time = (time.time() - stage2_start) * 1000
        print(f"\n[STAGE 2] ✅ Generated {len(candidates)} deliberations ({stage2_time:.0f}ms)")

        # ===== STAGE 3: FAST & CONSISTENT ALIGNMENT SCORING =====
        stage3_start = time.time()
        scored_candidates = self._stage3_score_deliberations(candidates)
        stage3_time = (time.time() - stage3_start) * 1000
        print(f"\n[STAGE 3] ✅ Scored {len(scored_candidates)} candidates ({stage3_time:.0f}ms)")
        for i, cand in enumerate(scored_candidates[:3], 1):
            print(f"   {i}. {cand['action']}: score={cand['alignment_score']:.3f}")

        # ===== STAGE 4: INFORMED FINAL SELECTION =====
        stage4_start = time.time()
        final_selection = self._stage4_final_selection(
            student_state=student_state,
            scored_candidates=scored_candidates,
            conversation_history=conversation_history
        )
        stage4_time = (time.time() - stage4_start) * 1000
        print(f"\n[STAGE 4] ✅ Final selection made ({stage4_time:.0f}ms)")
        print(f"   Selected: {final_selection['selected_action']}")
        print(f"   Alignment Score: {final_selection['alignment_score']:.3f}")

        # Calculate total metrics
        total_latency = (time.time() - start_time) * 1000

        # Count tokens (approximate)
        total_tokens = self._estimate_tokens(candidates, final_selection)

        # Build comprehensive result
        result = {
            # Turn info
            "turn_number": turn_number,
            "selected_action": final_selection['selected_action'],
            "deliberation": final_selection['deliberation'],
            "reasoning": final_selection['synthesis_reasoning'],
            "alignment_score": final_selection['alignment_score'],

            # Stage details
            "student_state": student_state,
            "all_candidates": scored_candidates,

            # Stage timings
            "stage1_time_ms": stage1_time,
            "stage2_time_ms": stage2_time,
            "stage3_time_ms": stage3_time,
            "stage4_time_ms": stage4_time,

            # EFFICIENCY METRICS
            "latency_ms": total_latency,
            "total_tokens": total_tokens,

            # EFFECTIVENESS METRIC
            "pedagogical_alignment_score": final_selection['alignment_score'],

            # TRANSPARENCY METRICS (for DAC, PCF)
            "deliberation_text": final_selection['deliberation'],
            "principle_references": self._extract_principle_references(final_selection['deliberation']),

            # Metadata
            "timestamp": time.time(),
            "planner_type": "experimental"
        }

        # Log
        self.turn_logs.append(result)
        self.student_state_history.append(student_state)

        print(f"\n⏱️  Total time: {total_latency/1000:.1f}s")
        print(f"{'='*80}\n")

        return result

    def _stage1_assess_state(self, student_message: str) -> Dict[str, float]:
        """
        STAGE 1: Structured State Assessment.
        Uses trained student_state_classifier.
        """
        if self.student_state_classifier is None:
            # Fallback heuristic
            msg_lower = student_message.lower()

            is_frustrated = any(word in msg_lower for word in 
                ['frustrated', 'stuck', 'can\'t', 'impossible', 'giving up'])
            is_confused = any(word in msg_lower for word in 
                ['confused', 'what', 'how', 'why', '?', 'don\'t understand'])
            is_correct = any(word in msg_lower for word in 
                ['got it', 'understand', 'makes sense', 'i see', 'right'])

            return {
                'is_frustrated': 0.8 if is_frustrated else 0.2,
                'is_confused': 0.7 if is_confused else 0.3,
                'is_correct': 0.7 if is_correct else 0.3,
                'is_disengaged': 0.2,
                'is_neutral': 0.5
            }
        else:
            # Use trained classifier - returns pedagogical state string
            pedagogical_state = self.student_state_classifier.predict(student_message)

            # Convert to probability dict
            state_probs = {
                'is_frustrated': 0.0,
                'is_confused': 0.0,
                'is_correct': 0.0,
                'is_disengaged': 0.0,
                'is_neutral': 0.0
            }

            # Set predicted state to high probability
            state_key = f"is_{pedagogical_state}"
            if state_key in state_probs:
                state_probs[state_key] = 0.9

            # Set others to low baseline
            for key in state_probs:
                if key != state_key:
                    state_probs[key] = 0.1

            return state_probs

    def _stage2_generate_deliberations(
        self,
        student_state: Dict[str, float],
        conversation_history: List[Dict[str, str]],
        current_problem: str,
        latest_student_msg: str,
        num_candidates: int,
        max_new_tokens: int,
        temperature: float
    ) -> List[Dict[str, str]]:
        """
        STAGE 2: Principled Deliberation & Candidate Generation.
        Uses DPO-finetuned deliberation generator.
        """
        candidates = []

        # Generate deliberations for multiple candidate actions
        for action in np.random.choice(self.pedagogical_actions, 
                                       size=min(num_candidates, len(self.pedagogical_actions)), 
                                       replace=False):

            # Build prompt for this candidate
            prompt = self._build_deliberation_prompt(
                student_state=student_state,
                current_problem=current_problem,
                latest_student_msg=latest_student_msg,
                candidate_action=action
            )

            # Generate deliberation using DPO-finetuned model
            deliberation = self._generate_single_deliberation(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            candidates.append({
                'action': action,
                'deliberation': deliberation
            })

        return candidates

    def _build_deliberation_prompt(
        self,
        student_state: Dict[str, float],
        current_problem: str,
        latest_student_msg: str,
        candidate_action: str
    ) -> str:
        """Build prompt for deliberation generation (same format as training)."""

        # Format student state
        state_str = self._format_student_state(student_state)

        prompt = f"""You are a pedagogical AI tutor for mathematics (Algebra, Grades 5-12).

STUDENT STATE: {state_str}
MATH PROBLEM: {current_problem}
STUDENT UTTERANCE: {latest_student_msg}
TUTOR ACTION: {candidate_action}

Generate a pedagogical deliberation (2-3 sentences) explaining WHY this action is appropriate. Reference specific pedagogical principles by number:
1. Foster Constructivism & Scaffolding
2. Manage Cognitive Load
3. Maintain Desirable Difficulty
4. Promote Metacognition
5. Foster Positive Affect & Validation
6. Maintain Factual Integrity

DELIBERATION:"""

        return prompt

    def _generate_single_deliberation(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float
    ) -> str:
        """Generate a single deliberation using DPO-finetuned model."""

        # Tokenize
        inputs = self.delib_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.delib_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.delib_tokenizer.pad_token_id,
                eos_token_id=self.delib_tokenizer.eos_token_id
            )

        # Decode
        full_output = self.delib_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the deliberation (after "DELIBERATION:")
        if "DELIBERATION:" in full_output:
            deliberation = full_output.split("DELIBERATION:")[-1].strip()
        else:
            deliberation = full_output[len(prompt):].strip()

        return deliberation

    def _stage3_score_deliberations(
        self,
        candidates: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        STAGE 3: Fast & Consistent Alignment Scoring.
        Uses trained alignment_scorer.
        """
        scored_candidates = []

        for candidate in candidates:
            # Generate embedding
            embedding = self.embedding_model.encode([candidate['deliberation']])

            # Predict alignment score
            score = self.alignment_scorer.predict(embedding)[0]
            score = float(np.clip(score, 0.0, 1.0))

            scored_candidates.append({
                'action': candidate['action'],
                'deliberation': candidate['deliberation'],
                'alignment_score': score
            })

        # Sort by alignment score (highest first)
        scored_candidates.sort(key=lambda x: x['alignment_score'], reverse=True)

        return scored_candidates

    def _stage4_final_selection(
        self,
        student_state: Dict[str, float],
        scored_candidates: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        STAGE 4: Informed Final Selection.
        Synthesizing step that considers all evidence.
        
        For now: Simply select the highest-scored candidate.
        Future: Could use LLM to synthesize and make nuanced final decision.
        """
        # Simple strategy: Pick highest alignment score
        best_candidate = scored_candidates[0]

        return {
            'selected_action': best_candidate['action'],
            'deliberation': best_candidate['deliberation'],
            'alignment_score': best_candidate['alignment_score'],
            'synthesis_reasoning': f"Selected {best_candidate['action']} with highest alignment score ({best_candidate['alignment_score']:.3f})"
        }

    def _format_student_state(self, student_state: Dict[str, float]) -> str:
        """Format student state for prompt."""
        states = []
        for key, value in student_state.items():
            if value > 0.5:
                state_name = key.replace('is_', '')
                states.append(f"{state_name} ({value:.2f})")

        return ", ".join(states) if states else "neutral"

    def _extract_principle_references(self, deliberation_text: str) -> List[int]:
        """Extract which principles are referenced in the deliberation."""
        references = []

        for i in range(1, 7):
            if f"Principle {i}" in deliberation_text or f"principle {i}" in deliberation_text:
                references.append(i)

        return references

    def _estimate_tokens(self, candidates: List[Dict], final_selection: Dict) -> int:
        """Estimate total tokens used across all stages."""
        # Rough estimate
        total_chars = sum(len(c['deliberation']) for c in candidates)
        total_chars += len(final_selection['synthesis_reasoning'])
        return total_chars // 4  # Rough token estimate

    def _get_latest_student_message(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract the most recent student message."""
        for turn in reversed(conversation_history):
            if turn["role"] == "student":
                return turn["message"]
        return "No student message yet."

    def calculate_narr(self) -> float:
        """Calculate Negative Affect Reduction Rate (NARR)."""
        if len(self.student_state_history) < 2:
            return 0.0

        triggers = []
        resolutions = []

        for i in range(len(self.student_state_history) - 1):
            state = self.student_state_history[i]

            if state.get('is_frustrated', 0) > 0.7:
                triggers.append(i)

                if i + 1 < len(self.student_state_history):
                    next_state = self.student_state_history[i + 1]
                    reduction = state['is_frustrated'] - next_state.get('is_frustrated', 0)
                    if reduction > 0.3:
                        resolutions.append(i)

        if len(triggers) == 0:
            return 1.0

        return len(resolutions) / len(triggers)

    def calculate_dac(self) -> float:
        """
        Calculate Deliberation-Action Congruence (DAC).
        Average alignment score of all selected actions.
        """
        if not self.turn_logs:
            return 0.0

        scores = [log['alignment_score'] for log in self.turn_logs]
        return np.mean(scores)

    def calculate_pcf(self) -> Dict[int, float]:
        """
        Calculate Principle Coverage Frequency (PCF).
        Percentage of deliberations referencing each principle.
        """
        if not self.turn_logs:
            return {}

        principle_counts = {i: 0 for i in range(1, 7)}
        total = len(self.turn_logs)

        for log in self.turn_logs:
            for principle_num in log.get('principle_references', []):
                principle_counts[principle_num] += 1

        # Convert to percentages
        pcf = {i: (count / total) * 100 for i, count in principle_counts.items()}

        return pcf

    def get_logs(self) -> List[Dict[str, Any]]:
        """Return all turn logs."""
        return self.turn_logs

    def save_logs(self, output_path: str):
        """Save logs to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.turn_logs, f, indent=2)
        print(f"✅ Logs saved to: {output_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.turn_logs:
            return {}

        # EFFICIENCY METRICS
        latencies = [log["latency_ms"] for log in self.turn_logs]
        tokens = [log["total_tokens"] for log in self.turn_logs]

        # Stage timings
        stage1_times = [log["stage1_time_ms"] for log in self.turn_logs]
        stage2_times = [log["stage2_time_ms"] for log in self.turn_logs]
        stage3_times = [log["stage3_time_ms"] for log in self.turn_logs]
        stage4_times = [log["stage4_time_ms"] for log in self.turn_logs]

        # EFFECTIVENESS METRICS
        alignment_scores = [log["alignment_score"] for log in self.turn_logs]

        # TRANSPARENCY METRICS
        dac = self.calculate_dac()
        pcf = self.calculate_pcf()

        # ROBUSTNESS METRICS
        narr = self.calculate_narr()

        return {
            # Basic info
            "total_turns": len(self.turn_logs),
            "planner_type": "experimental",

            # EFFICIENCY METRICS
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "avg_tokens_per_turn": np.mean(tokens),
            "total_tokens_used": sum(tokens),

            # Stage breakdown
            "avg_stage1_ms": np.mean(stage1_times),
            "avg_stage2_ms": np.mean(stage2_times),
            "avg_stage3_ms": np.mean(stage3_times),
            "avg_stage4_ms": np.mean(stage4_times),

            # EFFECTIVENESS METRICS
            "avg_pedagogical_alignment_score": np.mean(alignment_scores),
            "std_pedagogical_alignment_score": np.std(alignment_scores),

            # TRANSPARENCY METRICS
            "deliberation_action_congruence": dac,
            "principle_coverage_frequency": pcf,

            # ROBUSTNESS METRICS
            "negative_affect_reduction_rate": narr
        }


# Example usage and testing
if __name__ == "__main__":
    """Test the Experimental Planner."""

    print("\n🚀 Initializing Experimental Planner...\n")

    # Initialize planner with all trained models
    planner = ExperimentalPlanner(
        student_state_classifier_path="models/checkpoints/sec_mapper/best_model.pt",  # ← ADDED THIS
        deliberation_generator_path=r"C:\Users\Shlok\Downloads\final_model",
        alignment_scorer_path=r"models\alignment_scorer\alignment_scorer.pkl",
        use_quantization=True
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
        turn_number=1,
        num_candidates=5
    )

    print(f"\n📊 FINAL DECISION:")
    print(f"   Selected Action: {result['selected_action']}")
    print(f"   Alignment Score: {result['alignment_score']:.3f}")
    print(f"   Deliberation: {result['deliberation'][:200]}...")
    print(f"   Principles Referenced: {result['principle_references']}")

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
        turn_number=2,
        num_candidates=5
    )

    print(f"\n📊 FINAL DECISION:")
    print(f"   Selected Action: {result2['selected_action']}")
    print(f"   Alignment Score: {result2['alignment_score']:.3f}")

    # Save logs
    planner.save_logs("logs/experimental_planner/test_run.json")

    # Print comprehensive metrics
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE METRICS SUMMARY:")
    print("=" * 80)
    metrics = planner.get_metrics_summary()

    print(f"\n🎯 EFFICIENCY METRICS:")
    print(f"   Total Turns: {metrics['total_turns']}")
    print(f"   Avg Latency: {metrics['avg_latency_ms']:.0f} ms (±{metrics['std_latency_ms']:.0f})")
    print(f"   - Stage 1 (State): {metrics['avg_stage1_ms']:.0f} ms")
    print(f"   - Stage 2 (Delib): {metrics['avg_stage2_ms']:.0f} ms")
    print(f"   - Stage 3 (Score): {metrics['avg_stage3_ms']:.0f} ms")
    print(f"   - Stage 4 (Select): {metrics['avg_stage4_ms']:.0f} ms")
    print(f"   Avg Tokens/Turn: {metrics['avg_tokens_per_turn']:.0f}")

    print(f"\n✅ EFFECTIVENESS METRICS:")
    print(f"   Avg Alignment Score: {metrics['avg_pedagogical_alignment_score']:.3f} (±{metrics['std_pedagogical_alignment_score']:.3f})")

    print(f"\n🔍 TRANSPARENCY METRICS:")
    print(f"   Deliberation-Action Congruence (DAC): {metrics['deliberation_action_congruence']:.3f}")
    print(f"   Principle Coverage Frequency (PCF):")
    for principle_num, percentage in metrics['principle_coverage_frequency'].items():
        print(f"      Principle {principle_num}: {percentage:.1f}%")

    print(f"\n💪 ROBUSTNESS METRICS:")
    print(f"   Negative Affect Reduction Rate (NARR): {metrics['negative_affect_reduction_rate']:.2%}")

    print("\n" + "=" * 80)
    print("✅ EXPERIMENTAL PLANNER TEST COMPLETE")
    print("=" * 80)