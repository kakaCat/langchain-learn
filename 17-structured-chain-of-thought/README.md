# 17思维链 (Structured Chain of Thought - DeepSeek-R1 Traces)

This directory contains an implementation of the **DeepSeek-R1 Traces** structured reasoning framework. This framework is designed to handle high-complexity tasks by breaking down the reasoning process into four distinct stages.

## The 4 Stages

1.  **Problem Definition Phase (问题定义阶段)**:
    *   **Goal**: Rephrase the core task and calibrate attention.
    *   **Action**: Clarify the user's request, identify constraints, and define the problem precisely (e.g., "IMO math problems need to clarify geometric properties first").

2.  **Bloom Cycle (绽放周期)**:
    *   **Goal**: Preliminary decomposition and hypothesis generation.
    *   **Action**: Break down the problem and generate multiple hypotheses or approaches (e.g., "For code debugging, first assume variable type error, then check loop boundaries").

3.  **Rumination Cycle (重构周期)**:
    *   **Goal**: Self-verification and correction.
    *   **Action**: Critically evaluate the chosen approach, find flaws, and refine the plan (e.g., "Found formula derivation error, recalculate limit using L'Hopital's rule").

4.  **Final Decision Phase (最终决策阶段)**:
    *   **Goal**: Output conclusion with confidence.
    *   **Action**: Synthesize the final answer and provide a confidence score (e.g., "Proof gets 7 points, key step is auxiliary line construction").

## File Structure

*   `deepseek_r1_traces.py`: The Python implementation using LangChain to orchestrate the 4 stages using an LLM.

## Usage

To run the demo:

```bash
python deepseek_r1_traces.py
```

You can modify the `task` variable in the `__main__` block of `deepseek_r1_traces.py` to test different complex scenarios.
