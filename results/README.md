# InvaR1ant Evaluation Framework

## Overview

The **InvaR1ant Evaluation Framework** evaluates the ability of Large Language Models (LLMs) to generate constraints for worst-case execution paths in programs. It integrates with the OpenAI API for generating responses and uses the Z3 SMT solver to validate the logical equivalence of these responses against ground truth constraints. The framework also appends structured instructions to benchmark questions to ensure consistent and parseable outputs from the models.

## Features

- **Benchmark Dataset Integration**: The framework uses the `dannkoh/invaR1ant-benchmark` dataset to provide a standardized set of questions for evaluation.
- **Multi-Model Support**: Supports evaluation of multiple LLMs, including OpenAI's GPT models (`gpt-4o-mini-2024-07-18`, `gpt-4o-2024-11-20`, and `gpt-3.5-turbo-0125`).
- **Batch Processing**: Automatically generates and uploads batch requests to the OpenAI API for efficient evaluation.
- **Z3 SMT Solver Integration**: Validates the logical equivalence of generated constraints using the Z3 SMT solver.
- **Customizable Instructions**: Appends a specific instruction to each benchmark question, guiding the LLMs to format their responses in a structured manner.

## Workflow

1. **Dataset Loading**: The benchmark dataset is loaded using the `datasets` library and converted into a pandas DataFrame for processing.
2. **Question Formatting**: Each benchmark question is appended with an instruction to structure the response. The instruction specifies:
   - Thought processes should be enclosed in `<think></think>` tags.
   - Final answers should be enclosed in `<answer></answer>` tags.
3. **Batch Creation**: For each model, a batch of requests is created and saved as a `.jsonl` file. These batches are uploaded to the OpenAI API for processing.
4. **Response Validation**:
   - Responses are extracted and parsed to isolate the final answer.
   - The Z3 SMT solver is used to check the logical equivalence of the generated constraints against the ground truth.
5. **Result Aggregation**: Results are aggregated, including accuracy, number of correct and incorrect responses, and reasons for failures.

## Instructions for Benchmark Questions

To ensure consistent and structured responses from the LLMs, the following instruction is appended to each benchmark question:
```
Structure your response by placing your thought process between <think></think> tags and your final answer between <answer></answer> tags like this:\n<think>[REASONING]</think><answer>(assert (and (not ( = cell_0_0 0)) (not ( = cell_0_0 1))))</answer>
```
This instruction guides the LLMs to clearly delineate their reasoning process and final answer, making it easier to parse and validate the responses.