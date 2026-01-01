"""
Parse Control Results and Display Complete Metrics
Extracts IMV, Final Mastery, Effectiveness Scores from control_results.json
"""

import json
import numpy as np
from pathlib import Path


def parse_control_results(results_path: str = "experiments/evaluation_results/control_results.json"):
    """Parse control results and display comprehensive metrics."""
    
    print("\n" + "=" * 80)
    print("📊 DETAILED CONTROL PLANNER METRICS ANALYSIS")
    print("=" * 80)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\nTotal Scenarios: {len(results)}")
    print("=" * 80)
    
    # Initialize metric collectors
    all_metrics = {
        'imvs': [],
        'final_masteries': [],
        'initial_masteries': [],
        'effectiveness_scores': [],
        'original_effectiveness_scores': [],
        'latencies': [],
        'tokens': [],
        'narrs': [],
        'total_turns': []
    }
    
    # Collect metrics from each scenario
    for result in results:
        metrics = result.get('metrics', {})
        
        # Knowledge Tracing Metrics
        if 'imv' in metrics and metrics['imv'] != 0.0:
            all_metrics['imvs'].append(metrics['imv'])
        
        if 'final_mastery' in metrics and metrics['final_mastery'] > 0:
            all_metrics['final_masteries'].append(metrics['final_mastery'])
        
        # Get initial mastery from trajectory
        trajectory = metrics.get('mastery_trajectory', [])
        if trajectory:
            all_metrics['initial_masteries'].append(trajectory[0])
        
        # Effectiveness Scores (from turn results)
        turn_results = result.get('turn_results', [])
        for turn in turn_results:
            if 'effectiveness_score' in turn:
                all_metrics['effectiveness_scores'].append(turn['effectiveness_score'])
            if 'original_effectiveness_score' in turn:
                all_metrics['original_effectiveness_scores'].append(turn['original_effectiveness_score'])
        
        # Other metrics
        if 'avg_latency_ms' in metrics:
            all_metrics['latencies'].append(metrics['avg_latency_ms'])
        if 'avg_tokens_per_turn' in metrics:
            all_metrics['tokens'].append(metrics['avg_tokens_per_turn'])
        if 'negative_affect_reduction_rate' in metrics:
            all_metrics['narrs'].append(metrics['negative_affect_reduction_rate'])
        if 'total_turns' in metrics:
            all_metrics['total_turns'].append(metrics['total_turns'])
    
    # Print comprehensive summary
    print("\n LEARNING EFFECTIVENESS METRICS:")
    print("-" * 80)
    
    if all_metrics['imvs']:
        print(f"📈 Inferred Mastery Velocity (IMV):")
        print(f"   Mean:   {np.mean(all_metrics['imvs']):.4f}")
        print(f"   Median: {np.median(all_metrics['imvs']):.4f}")
        print(f"   Std:    {np.std(all_metrics['imvs']):.4f}")
        print(f"   Min:    {np.min(all_metrics['imvs']):.4f}")
        print(f"   Max:    {np.max(all_metrics['imvs']):.4f}")
    else:
        print(f"📈 IMV: Not calculated")
    
    print()
    
    if all_metrics['final_masteries']:
        print(f"🎯 Final Mastery:")
        print(f"   Mean:   {np.mean(all_metrics['final_masteries']):.3f}")
        print(f"   Median: {np.median(all_metrics['final_masteries']):.3f}")
        print(f"   Std:    {np.std(all_metrics['final_masteries']):.3f}")
        print(f"   Min:    {np.min(all_metrics['final_masteries']):.3f}")
        print(f"   Max:    {np.max(all_metrics['final_masteries']):.3f}")
    else:
        print(f"🎯 Final Mastery: Not calculated")
    
    print()
    
    if all_metrics['initial_masteries']:
        print(f"📊 Initial Mastery:")
        print(f"   Mean:   {np.mean(all_metrics['initial_masteries']):.3f}")
    
    print()
    
    if all_metrics['imvs'] and all_metrics['final_masteries'] and all_metrics['initial_masteries']:
        mastery_gain = np.mean(all_metrics['final_masteries']) - np.mean(all_metrics['initial_masteries'])
        print(f"📈 Average Mastery Gain: {mastery_gain:.3f}")
        print(f"   ({np.mean(all_metrics['initial_masteries']):.3f} → {np.mean(all_metrics['final_masteries']):.3f})")
    
    print("\n" + "=" * 80)
    print("⚙️ ACTION EFFECTIVENESS METRICS:")
    print("-" * 80)
    
    if all_metrics['effectiveness_scores']:
        print(f"🎯 Effectiveness Scores (After Reduction):")
        print(f"   Mean:   {np.mean(all_metrics['effectiveness_scores']):.2f}")
        print(f"   Median: {np.median(all_metrics['effectiveness_scores']):.2f}")
        print(f"   Std:    {np.std(all_metrics['effectiveness_scores']):.2f}")
        print(f"   Min:    {np.min(all_metrics['effectiveness_scores']):.2f}")
        print(f"   Max:    {np.max(all_metrics['effectiveness_scores']):.2f}")
    else:
        print(f"🎯 Effectiveness Scores: Not available")
    
    print()
    
    if all_metrics['original_effectiveness_scores']:
        print(f"📊 Original Effectiveness Scores (Before Reduction):")
        print(f"   Mean:   {np.mean(all_metrics['original_effectiveness_scores']):.2f}")
        print(f"   Median: {np.median(all_metrics['original_effectiveness_scores']):.2f}")
        print(f"   Std:    {np.std(all_metrics['original_effectiveness_scores']):.2f}")
        print(f"   Min:    {np.min(all_metrics['original_effectiveness_scores']):.2f}")
        print(f"   Max:    {np.max(all_metrics['original_effectiveness_scores']):.2f}")
    
    print("\n" + "=" * 80)
    print("⚡ EFFICIENCY METRICS:")
    print("-" * 80)
    
    if all_metrics['latencies']:
        print(f"⏱️  Latency:")
        print(f"   Mean:   {np.mean(all_metrics['latencies'])/1000:.1f}s")
        print(f"   Median: {np.median(all_metrics['latencies'])/1000:.1f}s")
    
    if all_metrics['tokens']:
        print(f"🔤 Tokens per Turn:")
        print(f"   Mean:   {np.mean(all_metrics['tokens']):.0f}")
        print(f"   Median: {np.median(all_metrics['tokens']):.0f}")
    
    if all_metrics['total_turns']:
        print(f"🔄 Turns per Scenario:")
        print(f"   Mean:   {np.mean(all_metrics['total_turns']):.1f}")
        print(f"   Total:  {sum(all_metrics['total_turns'])}")
    
    print("\n" + "=" * 80)
    print("💪 ROBUSTNESS METRICS:")
    print("-" * 80)
    
    if all_metrics['narrs']:
        print(f"😊 Negative Affect Reduction Rate (NARR):")
        print(f"   Mean:   {np.mean(all_metrics['narrs']):.2%}")
        print(f"   Median: {np.median(all_metrics['narrs']):.2%}")
    
    print("\n" + "=" * 80)
    print("📋 DETAILED BREAKDOWN:")
    print("=" * 80)
    
    # Show distribution of final mastery
    if all_metrics['final_masteries']:
        print(f"\n🎯 Final Mastery Distribution:")
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(all_metrics['final_masteries'], bins=bins)
        
        for i in range(len(bins)-1):
            count = hist[i]
            pct = (count / len(all_metrics['final_masteries'])) * 100
            bar = '█' * int(pct / 2)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Show distribution of IMV
    if all_metrics['imvs']:
        print(f"\n📈 IMV Distribution:")
        
        # Count negative, zero, and positive IMVs
        negative = sum(1 for imv in all_metrics['imvs'] if imv < 0)
        zero = sum(1 for imv in all_metrics['imvs'] if imv == 0)
        positive = sum(1 for imv in all_metrics['imvs'] if imv > 0)
        
        print(f"   Negative IMV: {negative:3d} ({negative/len(all_metrics['imvs'])*100:5.1f}%) - Learning regression")
        print(f"   Zero IMV:     {zero:3d} ({zero/len(all_metrics['imvs'])*100:5.1f}%) - No progress")
        print(f"   Positive IMV: {positive:3d} ({positive/len(all_metrics['imvs'])*100:5.1f}%) - Learning progress")
        
        print(f"\n   IMV Bins:")
        bins = [-0.1, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
        hist, _ = np.histogram(all_metrics['imvs'], bins=bins)
        
        for i in range(len(bins)-1):
            count = hist[i]
            pct = (count / len(all_metrics['imvs'])) * 100
            bar = '█' * int(pct / 2)
            print(f"   {bins[i]:.2f}-{bins[i+1]:.2f}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    
    return all_metrics


if __name__ == "__main__":
    # Run the analysis
    metrics = parse_control_results("experiments/evaluation_results/control_results.json")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 80)
    
    if metrics['imvs'] and metrics['final_masteries']:
        avg_imv = np.mean(metrics['imvs'])
        avg_final = np.mean(metrics['final_masteries'])
        
        print(f"1. Average learning velocity: {avg_imv:.4f} mastery/turn")
        print(f"2. Average final mastery: {avg_final:.3f} ({avg_final*100:.1f}%)")
    
    print("\n" + "=" * 80)