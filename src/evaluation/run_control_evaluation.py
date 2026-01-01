"""
Control Planner Evaluation Script
Evaluates ONLY the control planner (TRUE vanilla baseline).
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import torch
import gc

# Import planner
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.planners.control_planner import ControlPlanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControlEvaluator:
    """Evaluates Control planner only."""
    
    def __init__(
        self,
        test_scenarios_path: str,
        control_planner: ControlPlanner,
        output_dir: str = "experiments/evaluation_results"
    ):
        """Initialize evaluator."""
        self.test_scenarios_path = test_scenarios_path
        self.control_planner = control_planner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Result file paths
        self.control_results_path = self.output_dir / "control_results.json"
        self.progress_path = self.output_dir / "control_progress.json"
        
        # Load test scenarios
        with open(test_scenarios_path, 'r') as f:
            self.test_scenarios = json.load(f)
        
        logger.info(f"Loaded {len(self.test_scenarios)} test scenarios")
        
        # Results storage
        self.control_results = []
        
        # Progress tracking
        self.progress = {
            'control_completed': [],
            'last_updated': None
        }
        
        # Load existing results
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing results to support resume."""
        # Load progress
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r') as f:
                    self.progress = json.load(f)
                logger.info(f"📂 Loaded progress: {len(self.progress['control_completed'])} scenarios completed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        
        # Load control results
        if self.control_results_path.exists():
            try:
                with open(self.control_results_path, 'r') as f:
                    self.control_results = json.load(f)
                logger.info(f"📂 Loaded {len(self.control_results)} existing results")
            except Exception as e:
                logger.warning(f"Could not load results: {e}")
    
    def _save_progress(self):
        """Save current progress."""
        self.progress['last_updated'] = time.time()
        with open(self.progress_path, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _save_results(self):
        """Save results."""
        with open(self.control_results_path, 'w') as f:
            json.dump(self.control_results, f, indent=2)
    
    def _cleanup_cuda_memory(self):
        """Clean up CUDA memory to prevent leaks."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 CUDA memory cleaned")
    
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single scenario."""
        scenario_id = scenario['scenario_id']
        problem = scenario['problem']
        student_messages = scenario['student_messages']
        
        # Reset planner for new scenario
        self.control_planner.reset_for_new_scenario(problem)
        
        # Build conversation history turn by turn
        conversation_history = [
            {"role": "tutor", "message": f"Let's work on: {problem}"}
        ]
        
        turn_results = []
        turn_number = 0
        
        for turn_idx, student_turn in enumerate(student_messages, 1):
            turn_number = turn_idx
            conversation_history.append({
                "role": "student",
                "message": student_turn['message']
            })
            
            try:
                result = self.control_planner.plan_action(
                    conversation_history=conversation_history,
                    current_problem=problem,
                    turn_number=turn_idx,
                    max_new_tokens=1024,  # Reasonable for ToT reasoning
                    temperature=0.3       # Lower temp for structured output
                )
                
                turn_results.append(result)
                
                conversation_history.append({
                    "role": "tutor",
                    "message": f"[Tutor performs: {result['selected_action']}]"
                })
                
                # Periodic CUDA cleanup every 5 turns
                if turn_idx % 5 == 0:
                    self._cleanup_cuda_memory()
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error on scenario {scenario_id}, turn {turn_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Handle CUDA errors specifically
                if 'CUDA' in error_msg or 'out of memory' in error_msg.lower():
                    logger.warning(f"⚠️ CUDA error detected on scenario {scenario_id}, turn {turn_idx}")
                    self._cleanup_cuda_memory()
                    
                    # Return partial results for this scenario
                    metrics = self.control_planner.get_metrics_summary()
                    return {
                        "scenario_id": scenario_id,
                        "topic": scenario['topic'],
                        "difficulty": scenario['difficulty'],
                        "initial_student_state": scenario['initial_student_state'],
                        "num_turns": len(student_messages),
                        "completed_turns": len(turn_results),
                        "turn_results": turn_results,
                        "metrics": metrics,
                        "error": f"CUDA error at turn {turn_idx}",
                        "partial_completion": True
                    }
                
                # For other errors, continue to next turn
                continue
        
        # Get final metrics
        metrics = self.control_planner.get_metrics_summary()
        
        return {
            "scenario_id": scenario_id,
            "topic": scenario['topic'],
            "difficulty": scenario['difficulty'],
            "initial_student_state": scenario['initial_student_state'],
            "num_turns": len(student_messages),
            "completed_turns": len(turn_results),
            "turn_results": turn_results,
            "metrics": metrics,
            "partial_completion": False
        }
    
    def run_evaluation(self, max_scenarios: int = None):
        """Run evaluation."""
        scenarios_to_eval = self.test_scenarios[:max_scenarios] if max_scenarios else self.test_scenarios
        
        logger.info("=" * 80)
        logger.info("CONTROL PLANNER EVALUATION (VANILLA BASELINE)")
        logger.info("=" * 80)
        logger.info(f"Total scenarios: {len(scenarios_to_eval)}")
        logger.info("=" * 80)
        
        # Determine remaining scenarios
        completed_ids = set(self.progress['control_completed'])
        remaining_scenarios = [s for s in scenarios_to_eval if s['scenario_id'] not in completed_ids]
        
        if len(remaining_scenarios) == 0:
            logger.info("✅ All scenarios already completed!")
            self._print_summary()
            return
        
        logger.info(f"   Resuming: {len(remaining_scenarios)} scenarios remaining")
        
        for scenario in tqdm(remaining_scenarios, desc="Control Planner"):
            try:
                result = self.evaluate_scenario(scenario)
                
                self.control_results.append(result)
                self.progress['control_completed'].append(scenario['scenario_id'])
                
                # Save every 5 scenarios
                if len(self.progress['control_completed']) % 5 == 0:
                    self._save_results()
                    self._save_progress()
                    logger.info(f"💾 Progress saved: {len(self.progress['control_completed'])}/{len(scenarios_to_eval)}")
                    
                    # Aggressive CUDA cleanup every 5 scenarios
                    self._cleanup_cuda_memory()
                
                # Extra aggressive cleanup every 10 scenarios
                if len(self.progress['control_completed']) % 10 == 0:
                    logger.info("🧹 Performing deep memory cleanup...")
                    self._cleanup_cuda_memory()
                    time.sleep(2)  # Give system time to release memory
            
            except Exception as e:
                logger.error(f"Failed on scenario {scenario['scenario_id']}: {e}")
                import traceback
                traceback.print_exc()
                
                # Clean up and continue
                self._cleanup_cuda_memory()
                continue
        
        # Final save
        self._save_results()
        self._save_progress()
        logger.info(f"✅ Control planner evaluation complete!")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of results."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 CONTROL PLANNER SUMMARY")
        logger.info("=" * 80)
        
        if not self.control_results:
            logger.info("No results to summarize.")
            return
        
        # Aggregate metrics
        all_metrics = {
            'latencies': [],
            'tokens': [],
            'alignment_scores': [],
            'dacs': [],
            'narrs': []
        }
        
        # Track partial completions
        partial_completions = 0
        full_completions = 0
        
        for result in self.control_results:
            if result.get('partial_completion', False):
                partial_completions += 1
            else:
                full_completions += 1
            
            metrics = result.get('metrics', {})
            
            if 'avg_latency_ms' in metrics:
                all_metrics['latencies'].append(metrics['avg_latency_ms'])
            if 'avg_tokens_per_turn' in metrics:
                all_metrics['tokens'].append(metrics['avg_tokens_per_turn'])
            if 'avg_pedagogical_alignment_score' in metrics and metrics['avg_pedagogical_alignment_score'] > 0:
                all_metrics['alignment_scores'].append(metrics['avg_pedagogical_alignment_score'])
            if 'deliberation_action_congruence' in metrics and metrics['deliberation_action_congruence'] > 0:
                all_metrics['dacs'].append(metrics['deliberation_action_congruence'])
            if 'negative_affect_reduction_rate' in metrics:
                all_metrics['narrs'].append(metrics['negative_affect_reduction_rate'])
        
        logger.info(f"\nScenarios Evaluated: {len(self.control_results)}")
        logger.info(f"   Full Completions: {full_completions}")
        logger.info(f"   Partial Completions (CUDA errors): {partial_completions}")
        
        logger.info(f"\n🎯 EFFICIENCY:")
        if all_metrics['latencies']:
            logger.info(f"   Avg Latency: {np.mean(all_metrics['latencies']):.0f} ms")
        if all_metrics['tokens']:
            logger.info(f"   Avg Tokens/Turn: {np.mean(all_metrics['tokens']):.0f}")
        
        logger.info(f"\n✅ EFFECTIVENESS:")
        if all_metrics['alignment_scores']:
            logger.info(f"   Avg Alignment Score: {np.mean(all_metrics['alignment_scores']):.3f}")
        else:
            logger.info(f"   Avg Alignment Score: N/A (vanilla baseline)")
        
        logger.info(f"\n🔍 TRANSPARENCY:")
        if all_metrics['dacs']:
            logger.info(f"   Avg DAC: {np.mean(all_metrics['dacs']):.3f}")
        else:
            logger.info(f"   Avg DAC: N/A (vanilla baseline)")
        
        logger.info(f"\n💪 ROBUSTNESS:")
        if all_metrics['narrs']:
            logger.info(f"   Avg NARR: {np.mean(all_metrics['narrs']):.2%}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📁 Results saved: {self.control_results_path}")
        logger.info("=" * 80)


def main():
    """Main evaluation script."""
    
    logger.info("\n🚀 INITIALIZING CONTROL PLANNER EVALUATION...\n")
    
    # Clean CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Initial CUDA cleanup complete")
    
    # Initialize Control Planner (NO specialized models!)
    control_planner = ControlPlanner(
        use_quantization=True,
        enable_knowledge_tracing=True
        # NO alignment_scorer_path - this is vanilla baseline!
        # NO student_state_classifier_path
    )
    
    # Initialize Evaluator
    evaluator = ControlEvaluator(
        test_scenarios_path=r"C:\Users\shlok\Research\src\evaluation\test_scenarios.json",
        control_planner=control_planner,
        output_dir="experiments/evaluation_results"
    )
    
    # Run evaluation
    try:
        evaluator.run_evaluation(max_scenarios=None)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Evaluation interrupted by user")
        logger.info("Progress has been saved - you can resume later")
    except Exception as e:
        logger.error(f"Fatal error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Final CUDA cleanup complete")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CONTROL EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()