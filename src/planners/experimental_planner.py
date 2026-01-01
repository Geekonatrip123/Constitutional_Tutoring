"""
Experimental Planner: Deliberative Hybrid Architecture with Knowledge Tracing
4-stage pipeline 

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
import google.generativeai as genai

# Import Knowledge Tracer
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.evaluation.knowledge_tracer import KnowledgeTracer, LLMCorrectnessJudge

class RotatingGeminiWrapper:
    def __init__(self, planner):
        """
        Initialize wrapper with reference to planner.
        
        Args:
            planner: ExperimentalPlanner instance with Gemini rotation
        """
        self.planner = planner
    
    def generate_text(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """
        
        Args:
            prompt: Text prompt
            max_tokens: Max output tokens
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        max_retries = len(self.planner.gemini_api_keys)
        
        for attempt in range(max_retries):
            try:
                response = self.planner.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text.strip()
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if "429" in str(e) or "quota" in error_msg or "resource" in error_msg:
                    # Rate limit hit, rotate key
                    print(f"   [Classifier] API limit hit, rotating key...")
                    if self.planner._rotate_gemini_key():
                        time.sleep(1)
                        continue
                    else:
                        # All keys exhausted, use DPO fallback
                        print(f"   [Classifier] All keys exhausted, using DPO fallback")
                        return self.planner._enrich_with_qwen_fallback(prompt)
                else:
                    # Other error, raise immediately
                    raise
        
        
        return self.planner._enrich_with_qwen_fallback(prompt)
    
    def generate_json(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> dict:
        """
        Compatibility method for classifier that expects JSON output.
        
        Args:
            prompt: Text prompt
            max_tokens: Max output tokens
            temperature: Sampling temperature
        
        Returns:
            Parsed JSON dict
        """
        import json
        
        max_retries = len(self.planner.gemini_api_keys)
        
        for attempt in range(max_retries):
            try:
                response = self.planner.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                response_text = response.text.strip()
    
                try:
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"text": response_text}
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if "429" in str(e) or "quota" in error_msg or "resource" in error_msg:
                    # Rate limit hit, rotate key
                    print(f"   [Classifier] API limit hit, rotating key...")
                    if self.planner._rotate_gemini_key():
                        time.sleep(1)
                        continue
                    else:
                        # All keys exhausted, return fallback dict
                        print(f"   [Classifier] All keys exhausted, using fallback")
                        return {"text": "fallback", "error": "API exhausted"}
                else:
                    # Other error
                    if attempt == max_retries - 1:
                        # Last attempt, return error dict
                        return {"text": "error", "error": str(e)}
                    else:
                        time.sleep(1)
                        continue
        
        # All retries failed
        return {"text": "fallback", "error": "All retries failed"}


class ExperimentalPlanner:
    """
    Deliberative Hybrid Planner using 4-stage pipeline + Knowledge Tracing:
    Stage 0: COSIKE Scene Enrichment (Gemini 2.5 Flash API with DPO fallback)
    Stage 1: Student State Classification (student_state_classifier)
    Stage 2: Deliberation Generation (DPO-finetuned Qwen3-8B)
    Stage 3: Alignment Scoring (alignment_scorer)
    Stage 4: Final Selection (synthesizing LLM)
    + Knowledge Tracing: Track student mastery and calculate IMV (uses same DPO model)
    
    VRAM-optimized: Only ONE Qwen3-8B model loaded (DPO-finetuned)
    Used for: 1) Deliberations, 2) COSIKE fallback, 3) Knowledge tracing
    """

    def __init__(
        self,
        student_state_classifier_path: Optional[str] = None,
        deliberation_generator_path: str = None,
        alignment_scorer_path: str = None,
        use_quantization: bool = True,
        enable_knowledge_tracing: bool = True
    ):
        """
        Initialize Experimental Planner with all trained models.
        
        Args:
            student_state_classifier_path: Path to SEC best_model.pt
            deliberation_generator_path: Path to DPO-finetuned model (LoRA adapters)
            alignment_scorer_path: Path to alignment scorer
            use_quantization: Use 4-bit NF4 quantization
            enable_knowledge_tracing: Enable knowledge tracing with IMV metric
        """
        print("=" * 80)
        print("INITIALIZING EXPERIMENTAL PLANNER")
        print("4-Stage + COSIKE (Gemini 2.5 Flash + DPO Fallback) + Knowledge Tracing")
        print("VRAM-Optimized: Single Qwen3-8B (DPO-finetuned)")
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

        # Stage 0: COSIKE Enrichment with Gemini API (key rotation)
        self._init_gemini_api()

        # Stage 2: Deliberation Generator (DPO-finetuned) - LOAD BEFORE CLASSIFIER
        # This will be shared with COSIKE fallback and Knowledge Tracer
        self._load_deliberation_generator(deliberation_generator_path, use_quantization)

        # Stage 1: Student State Classifier (uses Gemini rotation wrapper)
        self._load_student_state_classifier(student_state_classifier_path)

        # Stage 3: Alignment Scorer
        self._load_alignment_scorer(alignment_scorer_path)

        # Knowledge Tracing - shares the SAME DPO model
        self.enable_knowledge_tracing = enable_knowledge_tracing
        self.knowledge_tracer = None
        self.llm_judge = None
        
        if enable_knowledge_tracing:
            print("\n[KNOWLEDGE TRACING] Initializing (sharing DPO model)...")
            try:
                # Share the DPO model with knowledge tracer
                self.llm_judge = LLMCorrectnessJudge(
                    model=self.delib_model,
                    tokenizer=self.delib_tokenizer,
                    use_shared_model=True
                )
                self.knowledge_tracer = KnowledgeTracer(llm_judge=self.llm_judge)
                print("   ✅ Knowledge Tracer initialized (shared model)")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not load knowledge tracer: {e}")
                import traceback
                traceback.print_exc()
                self.enable_knowledge_tracing = False

        # Logging
        self.turn_logs = []
        self.student_state_history = []
        self.current_problem = None

        print("\n✅ Experimental Planner initialized!")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   VRAM allocated: {allocated:.2f} GB")
        print("=" * 80)

    def _init_gemini_api(self):
        """Initialize Gemini API with multiple keys for rotation."""
        print(f"\n[STAGE 0] Initializing COSIKE Enrichment (Gemini 2.5 Flash)...")
        
        
        self.current_key_index = 0
        self.gemini_model_name = "gemini-2.5-flash"
        
        try:
            # Configure first key
            genai.configure(api_key=self.gemini_api_keys[self.current_key_index])
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
            
            print(f"   ✅ Gemini API initialized with {len(self.gemini_api_keys)} keys")
            print(f"   Model: {self.gemini_model_name}")
        except Exception as e:
            print(f"   ⚠️  Gemini initialization failed: {e}")
            print(f"   Will use DPO-Qwen fallback for all COSIKE enrichment")

    def _rotate_gemini_key(self):
        """Rotate to next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        
        try:
            genai.configure(api_key=self.gemini_api_keys[self.current_key_index])
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
            print(f"   🔄 Rotated to API key #{self.current_key_index + 1}")
            return True
        except Exception as e:
            print(f"   ⚠️  Key rotation failed: {e}")
            return False

    def _load_student_state_classifier(self, classifier_path: Optional[str]):
        """Load Stage 1: Student State Classifier with shared Gemini rotation."""
        print(f"\n[STAGE 1] Loading Student State Classifier...")

        self.student_state_classifier = None

        if classifier_path:
            try:
                from src.models.student_state_classifier import load_student_state_classifier

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                print(f"   Loading SEC model and k-NN classifier with balanced data...")
                print(f"   Using experimental planner's Gemini instance (15 keys with rotation)")

                # Create wrapper to share Gemini rotation
                gemini_wrapper = RotatingGeminiWrapper(self)

                self.student_state_classifier = load_student_state_classifier(
                    sec_checkpoint_path=classifier_path,
                    tokenizer_dir=str(Path(classifier_path).parent / "tokenizer"),
                    train_embeddings_path="src/data/processed/train_embeddings_balanced.npy",
                    train_labels_path="src/data/processed/train_labels_balanced.npy",
                    llm_api=gemini_wrapper,
                    device=device,
                    k=5
                )

                print(f"   ✅ Student state classifier loaded (using rotating Gemini + DPO fallback)")

            except Exception as e:
                print(f"   ⚠️  Warning: Could not load classifier: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Using heuristic fallback")
        else:
            print(f"   ⚠️  Using heuristic fallback (no classifier path provided)")

    def _load_deliberation_generator(self, generator_path: str, use_quantization: bool):
        """Load Stage 2: DPO-finetuned Deliberation Generator (ONLY QWEN MODEL)."""
        print(f"\n[STAGE 2] Loading Deliberation Generator (DPO-finetuned)...")

        if not generator_path:
            raise ValueError("deliberation_generator_path is required!")

        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(generator_path)
        BASE_MODEL = peft_config.base_model_name_or_path
        print(f"   Base model: {BASE_MODEL}")

        self.delib_tokenizer = AutoTokenizer.from_pretrained(
            generator_path,
            trust_remote_code=True
        )
        if self.delib_tokenizer.pad_token is None:
            self.delib_tokenizer.pad_token = self.delib_tokenizer.eos_token

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

        print(f"   Loading DPO LoRA adapters...")
        self.delib_model = PeftModel.from_pretrained(
            base_model,
            generator_path,
            is_trainable=False
        )
        self.delib_model.eval()
        print(f"   ✅ DPO-finetuned model loaded (shared for deliberation + COSIKE fallback + knowledge tracing)")

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

    def reset_for_new_scenario(self, problem: str):
        """Reset planner for a new scenario."""
        self.current_problem = problem
        self.student_state_history = []
        
        if self.knowledge_tracer:
            self.knowledge_tracer.reset()

    def _enrich_with_qwen_fallback(self, student_message: str) -> str:
        """
        Fallback COSIKE enrichment using DPO-finetuned Qwen3-8B.
        Used when all Gemini API keys are exhausted.
        
        Args:
            student_message: Original student message or prompt
        
        Returns:
            Enriched text with scene and keywords (or original message if it's a classifier prompt)
        """
        print(f"   🔄 Using DPO-Qwen for COSIKE enrichment (Gemini exhausted)")
        
        # Check if this is a classifier enrichment request (has "scene" keyword)
        if "scene" in student_message.lower() and "keywords" in student_message.lower():
            # This is the classifier trying to enrich - extract the actual student message
            # Parse out the student message from the prompt
            if 'Student: "' in student_message:
                actual_message = student_message.split('Student: "')[1].split('"')[0]
            else:
                actual_message = student_message
            
            # Just return the original message with basic enrichment
            return f"{actual_message}. Student is working through problem. thinking, processing, engaged"
        
        # Otherwise, this is a regular COSIKE enrichment request
        enrichment_prompt = f"""Analyze this student message and generate scene description + emotion keywords.

Student: "{student_message}"

Generate:
1. Scene: One sentence describing emotional/cognitive state
2. Keywords: 5 emotion keywords (confused, frustrated, confident, disengaged, neutral, uncertain, stuck, understanding, etc.)

Format:
Scene: [description]
Keywords: [word1, word2, word3, word4, word5]"""

        messages = [{"role": "user", "content": enrichment_prompt}]
        text = self.delib_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.delib_tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.delib_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.delib_tokenizer.pad_token_id,
                eos_token_id=self.delib_tokenizer.eos_token_id
            )
        
        response = self.delib_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse scene and keywords
        scene = ""
        keywords = ""
        
        try:
            if "Scene:" in response and "Keywords:" in response:
                parts = response.split("Keywords:")
                scene = parts[0].replace("Scene:", "").strip()
                keywords = parts[1].strip()
            else:
                # Fallback
                scene = "Student is working through problem"
                keywords = "thinking, processing, engaged"
        except:
            scene = "Student is working through problem"
            keywords = "thinking, processing, engaged"
        
        # Combine
        enriched_text = f"{student_message}. {scene}. {keywords}"
        
        return enriched_text

    def _enrich_with_cosike(self, student_message: str) -> str:
        """
        MANDATORY COSIKE scene enrichment using Gemini 2.5 Flash.
        Falls back to DPO-Qwen if all API keys exhausted.
        
        Args:
            student_message: Original student message
        
        Returns:
            Enriched text with scene and keywords
        """
        enrichment_prompt = f"""Analyze this student message and generate scene description + emotion keywords.

Student: "{student_message}"

Generate:
1. Scene: One sentence describing emotional/cognitive state
2. Keywords: 5 emotion keywords (confused, frustrated, confident, disengaged, neutral, uncertain, stuck, understanding, etc.)

Format:
Scene: [description]
Keywords: [word1, word2, word3, word4, word5]"""

        max_retries = len(self.gemini_api_keys)  # Try all keys
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(
                    enrichment_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1000
                    )
                )
                
                response_text = response.text.strip()
                
                # Parse scene and keywords
                scene = ""
                keywords = ""
                
                if "Scene:" in response_text and "Keywords:" in response_text:
                    parts = response_text.split("Keywords:")
                    scene = parts[0].replace("Scene:", "").strip()
                    keywords = parts[1].strip()
                else:
                    # Fallback
                    scene = "Student is working through problem"
                    keywords = "thinking, processing, engaged"
                
                # Combine
                enriched_text = f"{student_message}. {scene}. {keywords}"
                
                return enriched_text
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"   ⚠️  COSIKE enrichment attempt {attempt + 1} failed: {e}")
                
                if "429" in str(e) or "quota" in error_msg or "resource" in error_msg:
                    # Rate limit hit, try rotating key
                    if self._rotate_gemini_key():
                        time.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        # All keys exhausted, use DPO fallback
                        print(f"   🔄 All Gemini keys exhausted, switching to DPO-Qwen fallback")
                        return self._enrich_with_qwen_fallback(student_message)
                
                elif attempt == max_retries - 1:
                    # Final attempt failed, use DPO fallback
                    print(f"   🔄 Using DPO-Qwen fallback after {max_retries} attempts")
                    return self._enrich_with_qwen_fallback(student_message)
                else:
                    time.sleep(2)  # Retry with delay
        
        print(f"   🔄 Using DPO-Qwen fallback (final safety)")
        return self._enrich_with_qwen_fallback(student_message)

    def plan_action(
        self,
        conversation_history: List[Dict[str, str]],
        current_problem: str,
        turn_number: int,
        num_candidates: int = 5,
        max_new_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Main planning function - 4-stage deliberative pipeline + knowledge tracing."""
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"EXPERIMENTAL PLANNER - TURN {turn_number}")
        print(f"{'='*80}")

        if turn_number == 1 or self.current_problem != current_problem:
            self.reset_for_new_scenario(current_problem)

        latest_student_msg = self._get_latest_student_message(conversation_history)

        # ===== STAGE 0: COSIKE ENRICHMENT =====
        stage0_start = time.time()
        enriched_msg = self._enrich_with_cosike(latest_student_msg)
        stage0_time = (time.time() - stage0_start) * 1000
        print(f"\n[STAGE 0]  COSIKE enrichment (Gemini 2.5 Flash / DPO fallback) ({stage0_time:.0f}ms)")

        # ===== STAGE 1: STRUCTURED STATE ASSESSMENT =====
        stage1_start = time.time()
        student_state = self._stage1_assess_state(enriched_msg)  # Use enriched text!
        stage1_time = (time.time() - stage1_start) * 1000
        print(f"\n[STAGE 1]  State assessed ({stage1_time:.0f}ms)")
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

        # ===== KNOWLEDGE TRACING UPDATE =====
        mastery = None
        if self.enable_knowledge_tracing and self.knowledge_tracer:
            expected_state = self._get_primary_state(student_state)
            
            mastery = self.knowledge_tracer.update(
                problem=current_problem,
                student_message=latest_student_msg,
                expected_state=expected_state,
                tutor_action=final_selection['selected_action'],
                tutor_response=None
            )
            
            print(f"\n[KNOWLEDGE TRACING] Mastery updated: {mastery:.3f}")

        total_latency = (time.time() - start_time) * 1000
        total_tokens = self._estimate_tokens(candidates, final_selection)

        result = {
            "turn_number": turn_number,
            "selected_action": final_selection['selected_action'],
            "deliberation": final_selection['deliberation'],
            "reasoning": final_selection['synthesis_reasoning'],
            "alignment_score": final_selection['alignment_score'],
            "student_state": student_state,
            "all_candidates": scored_candidates,
            "stage0_time_ms": stage0_time,  # COSIKE
            "stage1_time_ms": stage1_time,
            "stage2_time_ms": stage2_time,
            "stage3_time_ms": stage3_time,
            "stage4_time_ms": stage4_time,
            "latency_ms": total_latency,
            "total_tokens": total_tokens,
            "pedagogical_alignment_score": final_selection['alignment_score'],
            "student_mastery": mastery,
            "deliberation_text": final_selection['deliberation'],
            "principle_references": self._extract_principle_references(final_selection['deliberation']),
            "timestamp": time.time(),
            "planner_type": "experimental"
        }

        self.turn_logs.append(result)
        self.student_state_history.append(student_state)

        print(f"\n⏱️  Total time: {total_latency/1000:.1f}s")
        print(f"{'='*80}\n")

        return result

    def _get_primary_state(self, student_state: Dict[str, float]) -> str:
        """Get the primary emotional state from probabilities."""
        max_prob = max(student_state.values())
        for key, value in student_state.items():
            if value == max_prob:
                return key.replace('is_', '')
        return "neutral"

    def _stage1_assess_state(self, enriched_message: str) -> Dict[str, float]:
        """
        STAGE 1: Structured State Assessment.
        Uses COSIKE-enriched text with trained classifier.
        """
        if self.student_state_classifier is None:
            msg_lower = enriched_message.lower()

            is_frustrated = any(word in msg_lower for word in 
                ['frustrated', 'stuck', 'can\'t', 'impossible', 'giving up'])
            is_confused = any(word in msg_lower for word in 
                ['confused', 'what', 'how', 'why', '?', 'don\'t understand', 'uncertain', 'lost'])
            is_correct = any(word in msg_lower for word in 
                ['got it', 'understand', 'makes sense', 'i see', 'right', 'confident', 'clarity'])

            return {
                'is_frustrated': 0.8 if is_frustrated else 0.2,
                'is_confused': 0.7 if is_confused else 0.3,
                'is_correct': 0.7 if is_correct else 0.3,
                'is_disengaged': 0.2,
                'is_neutral': 0.5
            }
        else:
            # Use enriched text for classification
            pedagogical_state = self.student_state_classifier.predict(enriched_message)

            state_probs = {
                'is_frustrated': 0.0,
                'is_confused': 0.0,
                'is_correct': 0.0,
                'is_disengaged': 0.0,
                'is_neutral': 0.0
            }

            state_key = f"is_{pedagogical_state}"
            if state_key in state_probs:
                state_probs[state_key] = 0.9

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
        """STAGE 2: Principled Deliberation & Candidate Generation."""
        candidates = []

        for action in np.random.choice(self.pedagogical_actions, 
                                       size=min(num_candidates, len(self.pedagogical_actions)), 
                                       replace=False):

            prompt = self._build_deliberation_prompt(
                student_state=student_state,
                current_problem=current_problem,
                latest_student_msg=latest_student_msg,
                candidate_action=action
            )

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
        """Build prompt for deliberation generation."""
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
        inputs = self.delib_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

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

        full_output = self.delib_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "DELIBERATION:" in full_output:
            deliberation = full_output.split("DELIBERATION:")[-1].strip()
        else:
            deliberation = full_output[len(prompt):].strip()

        return deliberation

    def _stage3_score_deliberations(
        self,
        candidates: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """STAGE 3: Fast & Consistent Alignment Scoring."""
        scored_candidates = []

        for candidate in candidates:
            embedding = self.embedding_model.encode([candidate['deliberation']])
            score = self.alignment_scorer.predict(embedding)[0]
            score = float(np.clip(score, 0.0, 1.0))

            scored_candidates.append({
                'action': candidate['action'],
                'deliberation': candidate['deliberation'],
                'alignment_score': score
            })

        scored_candidates.sort(key=lambda x: x['alignment_score'], reverse=True)
        return scored_candidates

    def _stage4_final_selection(
        self,
        student_state: Dict[str, float],
        scored_candidates: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """STAGE 4: Informed Final Selection."""
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
        total_chars = sum(len(c['deliberation']) for c in candidates)
        total_chars += len(final_selection['synthesis_reasoning'])
        return total_chars // 4

    def _get_latest_student_message(self, conversation_history: List[Dict[str, str]]) -> str:
        """Extract the most recent student message."""
        for turn in reversed(conversation_history):
            if turn["role"] == "student":
                return turn["message"]
        return "No student message yet."

    def calculate_imv(self) -> float:
        """Calculate Inferred Mastery Velocity (IMV) for current scenario."""
        if not self.enable_knowledge_tracing or not self.knowledge_tracer:
            return 0.0
        
        return self.knowledge_tracer.calculate_imv()

    def get_final_mastery(self) -> float:
        """Get final mastery level for current scenario."""
        if not self.enable_knowledge_tracing or not self.knowledge_tracer:
            return 0.0
        
        return self.knowledge_tracer.get_final_mastery()

    def get_mastery_trajectory(self) -> List[float]:
        """Get mastery trajectory over all turns."""
        if not self.enable_knowledge_tracing or not self.knowledge_tracer:
            return []
        
        return self.knowledge_tracer.get_mastery_trajectory()

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
        """Calculate Deliberation-Action Congruence (DAC)."""
        if not self.turn_logs:
            return 0.0

        scores = [log['alignment_score'] for log in self.turn_logs]
        return np.mean(scores)

    def calculate_pcf(self) -> Dict[int, float]:
        """Calculate Principle Coverage Frequency (PCF)."""
        if not self.turn_logs:
            return {}

        principle_counts = {i: 0 for i in range(1, 7)}
        total = len(self.turn_logs)

        for log in self.turn_logs:
            for principle_num in log.get('principle_references', []):
                principle_counts[principle_num] += 1

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
        """Get comprehensive metrics summary including IMV."""
        if not self.turn_logs:
            return {}

        latencies = [log["latency_ms"] for log in self.turn_logs]
        tokens = [log["total_tokens"] for log in self.turn_logs]

        stage0_times = [log["stage0_time_ms"] for log in self.turn_logs]  # COSIKE
        stage1_times = [log["stage1_time_ms"] for log in self.turn_logs]
        stage2_times = [log["stage2_time_ms"] for log in self.turn_logs]
        stage3_times = [log["stage3_time_ms"] for log in self.turn_logs]
        stage4_times = [log["stage4_time_ms"] for log in self.turn_logs]

        alignment_scores = [log["alignment_score"] for log in self.turn_logs]
        
        imv = self.calculate_imv()
        final_mastery = self.get_final_mastery()
        mastery_trajectory = self.get_mastery_trajectory()

        dac = self.calculate_dac()
        pcf = self.calculate_pcf()
        narr = self.calculate_narr()

        return {
            "total_turns": len(self.turn_logs),
            "planner_type": "experimental",
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "avg_tokens_per_turn": np.mean(tokens),
            "total_tokens_used": sum(tokens),
            "avg_stage0_ms": np.mean(stage0_times),  # COSIKE
            "avg_stage1_ms": np.mean(stage1_times),
            "avg_stage2_ms": np.mean(stage2_times),
            "avg_stage3_ms": np.mean(stage3_times),
            "avg_stage4_ms": np.mean(stage4_times),
            "avg_pedagogical_alignment_score": np.mean(alignment_scores),
            "std_pedagogical_alignment_score": np.std(alignment_scores),
            "imv": imv,
            "final_mastery": final_mastery,
            "mastery_trajectory": mastery_trajectory,
            "deliberation_action_congruence": dac,
            "principle_coverage_frequency": pcf,
            "negative_affect_reduction_rate": narr
        }