# DSA Compression Experiment

This directory contains experiments to validate the effectiveness of **DSA (DeepSeek Sparse Attention) Context Compression** with **Reasoning Traces**.

## Goal
Validate that retaining reasoning traces (`<think>...</think>`) during context compression allows the Agent to preserve "Meta-Cognition" (User Profile, Interaction Strategy), resulting in better performance on future tasks compared to standard summarization.

## Structure
- `experiment_runner.py`: The main script to run A/B tests comparing "Baseline" vs "DSA" strategies (Single Case).
- `statistical_validation.py`: A multi-scenario, statistical evaluation script using **LLM-as-a-Judge**.

## Configuration
The experiment can be configured using a `.env` file in this directory:

```env
OLLAMA_MODEL=deepseek-r1-32b-q4km:latest
OLLAMA_BASE_URL=http://localhost:11434
```

## How to Run

### 1. Single Case Demo
```bash
python experiment_runner.py
```
Runs a single conversation (Python Beginner) to visualize the difference between Baseline and DSA outputs.

### 2. Statistical Validation
```bash
python statistical_validation.py
```
Runs a batch of test scenarios (Beginner, Executive, JSON Strict, Philosophical) multiple times to calculate average scores.

## Experiment Results (Sample)
**Model**: DeepSeek-R1 32B
**Metric**: 1-10 Score by LLM Judge on "Context Adherence"

| Scenario | Baseline Score | DSA Score | Improvement |
|----------|---------------|-----------|-------------|
| Python Beginner | 8.00 | 8.00 | 0.00 |
| Busy Executive | 8.00 | 8.00 | 0.00 |
| **JSON Strict Output** | **1.00** | **10.00** | **+9.00** |
| Philosophical Debater | 9.00 | 8.00 | -1.00 |
| **Average** | **6.50** | **8.50** | **+30.77%** |

### Key Findings
1. **Constraint Preservation**: DSA shines in scenarios with strict format constraints (e.g., JSON only). The `[Interaction Strategy]` field explicitly captures these constraints, whereas standard summarization often loses them.
2. **General Adaptation**: For general style adaptation (Beginner/Executive), both Baseline and DSA perform well with a strong model like DeepSeek-R1 32B.
