

```
# Constitutional Tutoring: Deliberative Planning for Pedagogically-Aligned AI via RLAIF

[![Code](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/Geekonatrip123/Constitutional_Tutoring.git)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow.svg)](https://huggingface.co/SAMstark1235618317363/qwen3-8b-pedagogical-dpo)
[![Data](https://img.shields.io/badge/Data-Google_Drive-green.svg)](https://drive.google.com/drive/folders/1RnhrVWdetrm0UBmQVb6-jCflWzN4CE9k?usp=sharing)

Official implementation of **"Constitutional Tutoring: Deliberative Planning for Pedagogically-Aligned AI via RLAIF"** (ARR 2026).

---

## 🎯 Overview

Large Language Models excel at conversational fluency but struggle in pedagogical scenarios where **withholding information** is often more effective than immediate helpfulness. We introduce the **Deliberative Pedagogical Planner (DPP)**, a hybrid architecture that aligns AI tutoring behavior with a *Pedagogical Constitution* derived from learning science.

### Key Results
- **+286% learning velocity** (IMV: 0.080 vs 0.021, *p* < 0.001)
- **+53% final mastery** (0.724 vs 0.472, *p* < 0.001)
- **+168% affect management** (NARR: 95.0% vs 35.4%, *p* < 0.001)
- **93.9% deliberation-action congruence** (unique transparency metric)

**Critical Finding:** Both our system and vanilla ToT baseline self-rate their effectiveness similarly (~8/10), yet constitutional grounding produces **3.8× better learning outcomes** — proving that **generic helpfulness ≠ pedagogical effectiveness**.

---

## 🏗️ Architecture

![DPP Architecture](paper/figures/final_diagram.png)

The system employs a four-stage deliberation loop:

**Stage 0: COSIKE Enrichment**  
Augments student utterances with emotional context using Gemini 2.5 Flash + DPO-finetuned Qwen3-8B fallback.

**Stage 1: Student State Assessment**  
Siamese k-NN classifier (SEC) trained on 2,847 labeled utterances produces probability distributions over 5 affective states.

**Stage 2: Constitutional Deliberation Generation**  
DPO-finetuned Qwen3-8B generates 5 candidate actions with explicit principle-based reasoning (150-200 tokens each).

**Stage 3: Alignment Scoring**  
XGBoost regressor scores candidates by pedagogical validity (RMSE: 0.087).

**Stage 4: Final Selection**  
Executive LLM synthesizes evidence to select the optimal action.

**Parallel: Knowledge Tracing Module**  
Bayesian knowledge tracing with LLM-based correctness judgment tracks student mastery (IMV calculation).

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone YOUR_GITHUB_LINK
cd constitutional-tutoring

# Create virtual environment
conda create -n dpp python=3.10
conda activate dpp

# Install dependencies
pip install -r requirements.txt
```

### Download Models & Data

**Download DPO-finetuned Qwen3-8B Model:**
```bash
# Using HuggingFace CLI
huggingface-cli download SAMstark1235618317363/qwen3-8b-pedagogical-dpo --local-dir models/final_model

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SAMstark1235618317363/qwen3-8b-pedagogical-dpo",
    local_dir="models/final_model"
)
```

**Download Additional Models & Data:**
```bash
# Alignment scorer, SEC classifier, evaluation data
gdown YOUR_GOOGLE_DRIVE_LINK
unzip dpp_additional_models_data.zip
```

### Run Evaluation

**Evaluate Constitutional DPP:**
```bash
python src/evaluation/run_evaluation.py \
    --deliberation_model_path models/final_model \
    --alignment_scorer_path models/alignment_scorer \
    --student_state_classifier_path models/sec_classifier \
    --test_scenarios data/test_scenarios.json \
    --output_dir experiments/evaluation_results
```

**Evaluate Vanilla ToT Baseline:**
```bash
python src/evaluation/run_control_evaluation.py \
    --test_scenarios data/test_scenarios.json \
    --output_dir experiments/evaluation_results
```


---

## 🤖 Using the DPO-Finetuned Model

You can use our constitutional deliberation model directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_name = "SAMstark1235618317363/qwen3-8b-pedagogical-dpo"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Example: Generate constitutional deliberation
prompt = """Given student state: confused, frustrated (0.7)
Problem: Solve 3x + 7 = 22
Student message: "I don't know where to start..."

Generate pedagogical deliberation referencing constitutional principles:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
deliberation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(deliberation)
```

**Expected Output:**
```
Constitutional Deliberation:

P2 (Manage Cognitive Load) is highly relevant here - the student's 
frustration suggests cognitive overwhelm. Rather than providing a worked 
example (which would increase load), we should simplify the problem space.

P1 (Foster Constructivism) suggests asking a Socratic question to guide 
discovery: "What operation would help you isolate the x term?"

P5 (Foster Positive Affect) requires acknowledging their frustration 
before proceeding: "It's okay to feel stuck - let's break this down together."

Recommended Action: ask_socratic_question
Reasoning: Reduces cognitive load while maintaining student agency...
```

---

## 🎓 The Pedagogical Constitution

Our framework is grounded in six principles derived from learning science:

| Principle | Description | Citations |
|-----------|-------------|-----------|
| **P1: Foster Constructivism & Scaffolding** | Guide discovery rather than direct telling | Vygotsky (1978), Chi et al. (2001) |
| **P2: Manage Cognitive Load** | Prevent overwhelm by simplifying tasks | Sweller (1988) |
| **P3: Maintain Desirable Difficulty** | Keep students in zone of proximal development | Bjork (1994), VanLehn (2011) |
| **P4: Promote Metacognition** | Encourage reflection on thinking processes | Schraw & Dennison (1994) |
| **P5: Foster Positive Affect** | Validate emotions while maintaining challenge | D'Mello & Graesser (2012) |
| **P6: Maintain Factual Integrity** | Ensure mathematical correctness | Renkl et al. (1998) |

---

## 📊 Novel Evaluation Metrics

### Inferred Mastery Velocity (IMV)
Measures learning efficiency as the sum of positive mastery changes across all turns.
```python
imv = sum(mastery_changes) if mastery_changes > 0 else 0.0
```
**Higher is better** — indicates faster knowledge acquisition.

### Deliberation-Action Congruence (DAC)
Quantifies transparency by measuring alignment between generated deliberations and selected actions.
```python
dac = mean(alignment_scores_of_selected_actions)
```
**Unique to Constitutional DPP** — enables human verification of pedagogical reasoning.

### Negative Affect Reduction Rate (NARR)
Captures emotional scaffolding effectiveness.
```python
narr = resolved_frustration_episodes / total_frustration_episodes
```
**Higher is better** — shows ability to address student frustration/confusion.

---

## 🔬 Reproducing Paper Results

### Main Results Table
```bash
python experiments/analysis/extract_metrics.py
```

**Expected Output:**
```
================================================================================
📊 DETAILED METRICS ANALYSIS
================================================================================

🎓 LEARNING EFFECTIVENESS:
   IMV (Constitutional): 0.0800
   IMV (Vanilla ToT):    0.0207
   Improvement:          +286% (p < 0.001)

🎯 FINAL MASTERY:
   Constitutional DPP:   0.724
   Vanilla ToT:          0.472
   Improvement:          +53% (p < 0.001)

💪 AFFECT MANAGEMENT:
   NARR (Constitutional): 95.0%
   NARR (Vanilla ToT):    35.4%
   Improvement:           +168% (p < 0.001)
```

### Statistical Tests
All improvements are statistically significant at *p* < 0.001 using paired t-tests across 100 evaluation scenarios (576 total turns).

---

## 🛠️ Training Your Own DPO Model

If you want to replicate our DPO fine-tuning process:

```bash
python src/training/train_deliberation_generator.py \
    --base_model Qwen/Qwen3-8B \
    --training_data data/training_data/dpo_triplets.json \
    --output_dir models/my_dpo_model \
    --beta 0.1 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-6
```

See our [training documentation](docs/TRAINING.md) for details on:
- Generating synthetic DPO preference pairs
- Constitutional rubric for labeling
- Hyperparameter tuning

---





---

## 📧 Contact

For questions or collaboration inquiries:
- **Primary Contact:** shlok.sand@research.iiit.ac.in
- **Institution:** IIIT Hyderabad

---

## 🙏 Acknowledgments



## 📄 License

This project is licensed under the MIT License.

Models are released under the same license as the base Qwen model.

---

---

**Built with ❤️ for advancing AI education research**
```
