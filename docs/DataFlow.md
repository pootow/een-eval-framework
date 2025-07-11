# EEN Eval Framework Data Flow Documentation

## Overview

The EEN Eval framework follows a two-phase execution model with clear data structures flowing between components. This document outlines the complete data flow from configuration to final metrics.

## Phase 1: Inference

### Input Data Structures

#### 1. Configuration
```yaml
sample_params:
  temperature: 0.7
  max_tokens: 2000
  num_samples: 16  # Key parameter for pass@k metrics, should set to at least num_trials * k

models:
  - name: "model_name"
    type: "openai"
    config: {...}
  - name: "local_model" # config only name will default to local model runing at http://localhost:1234

dataset: "path/to/dataset.jsonl"
```

#### 2. Dataset Items
```python
DatasetItem = {
    "id": str,           # Unique identifier for the dataset item
    "data": dict,        # Item-specific data (question, problem, transcript, etc.)
    "ground_truth": Any, # Ground truth data in any format (list, dict, etc.)
                        # Structure depends on evaluation method requirements
    # Additional fields as needed
}
```

### Processing Flow

```
Dataset Item → Multiple Samples → Inference Results
     ↓              ↓                    ↓
   1 item    × num_samples     = num_samples results
```

### Output Data Structure

#### Inference Results (`responses.jsonl`)
```python
InferenceResult = {
    "item_id": str,              # Links back to original dataset item
                                 # "problem_id" is removed in favor of item_id
    "sample_id": str,            # Unique per sample: "{item_id}_sample_{index}"
    "sample_index": int,         # 0 to num_samples-1
    "total_samples": int,        # Total samples (= model count * items count * num_samples)
    "model_name": str,           # Model name (just a friendly display name,
                                 # for grouping use facets)
    "prompt": str,               # Processed prompt sent to model
    "response": str,             # Model's raw response
    "inference_time": float,     # Time taken for this inference
    "timestamp": float,          # When this inference was made
    "metadata": dict,            # Additional inference-specific metadata
                                 # e.g., model version, prompt template, sampling params, etc.
    # Error handling
    "error": str | None          # Error message if inference failed
}
```

## Phase 2: Evaluation

### Input: Inference Results
The evaluation phase reads the `responses.jsonl` file from Phase 1.

### Processing Flow

```
Inference Results → Evaluation Methods → Evaluation Results → Metrics → Final Results
       ↓                     ↓                     ↓
   1 result       ×    count of labels = num_evaluation_results
```

#### 1. Evaluation Methods
Each evaluation method processes one inference result into one or more labels.

```python
def custom_evaluation_method(
    response: str,           # From inference_result["response"]
    ground_truth: dict,      # From dataset["ground_truth"]
    inference_result: dict,  # Full inference result for context
    **params
) -> dict:
    """
    Returns:
    {
        "labels": [
            {
                "label": {"name": "label_name", "description": "optional"},
                "result": {
                    "passed": bool,      # Whether this evaluation passed
                    "score": float,      # Numeric score (0.0-1.0 typically)
                    "custom_fields": {   # Additional evaluation-specific data
                        "confidence": float,
                        "reasoning": str,
                        # ... other custom fields
                    }
                }
            },
            # ... more labels
        ],
        "custom_fields": { ... },  # Additional evaluation method metadata
    }
    """
```

#### 2. Evaluation Results
```python
EvaluationResult = {
    # Preserved from inference
    "item_id": str,              # Original dataset item
                                 # "problem_id" is removed in favor of item_id
    "sample_id": str,            # Unique sample identifier
    "sample_index": int,         # Sample number within item
    "label": str,                # Label name (e.g., "clearness", "correctness")
    "model_name": str,           # Model name (just a friendly display name,
                                 # for grouping use facets)
    
    # From evaluation method
    "passed": bool,              # Whether this sample passed (used by built-in metrics)
    "score": float,              # Numeric score (optional) (used by built-in metrics)
    "detailed_results": dict,    # Copy of "custom_fields" from evaluation method
                                 # And additional fields as needed
    
    # Metadata
    "evaluation_time": float,    # Time taken for evaluation
    "timestamp": float           # When evaluation was performed
    "metadata": dict,            # Additional inference-specific metadata
                                 # Additional evaluation-specific metadata (e.g., evaluation method config etc.)
}
```

### Metrics Calculation

Metrics are calculated based on the evaluation results. But to make metrics meaningful, we need to specify the facets. For example, a problem might solved by

- multiple models
  - the metric of different models matters
- multiple prompts
  - the metric of different prompts matters
- multiple models and prompts
  - both models and prompts matters

So we need to specify the fields (from inference result and evaluation result) that can be used to group the evaluation results.

If no facets are specified, the metrics will be calculated over all evaluation results as one single facet and you won't be able to distinguish between different models, prompts, etc.

#### Facets for Metrics config example

```yaml
metrics:
  # Built-in metrics
  - name: "pass@1"
    type: "pass_at_k"
    facets:
      - metadata.model_id
      - prompt_template
    params:
      k: 1
      num_trials: 16
      aggregation: "mean"
```

This will yield metrics (model count x prompt template count) like:
```jsonl
{"metadata.model_id": "model_1", "prompt_template": "template_1", "label": "exact_match", "pass_at_k": 0.91, "average_sample_count": 16, "total_sample_count": 320, ...}
{"metadata.model_id": "model_1", "prompt_template": "template_2", "label": "exact_match", "pass_at_k": 0.92, "average_sample_count": 16, "total_sample_count": 320, ...}
{"metadata.model_id": "model_2", "prompt_template": "template_1", "label": "exact_match", "pass_at_k": 0.93, "average_sample_count": 16, "total_sample_count": 320, ...}
{"metadata.model_id": "model_2", "prompt_template": "template_2", "label": "exact_match", "pass_at_k": 0.84, "average_sample_count": 16, "total_sample_count": 320, ...}
...(and so on for each model and prompt combination)
```

#### Data Grouping for Metrics

- Grouping is needed for all metrics. If no grouping, metrics will be calculated over all evaluation results as one single facet (which is obviously not useful).
- This grouping is generally done before calling the metric calculation function, that is each metric function will receive only one of these groups as input.

The following code is just an illustration. The grouping logic will implemented in the framework, not in the metric function itself.

```python
# Group evaluation results by facets (e.g., facet1, facet2) and label.
# (NOTE: each label will be treated as an extra facet **only when** grouping evaluation results for calculating metrics, besides grouping, label will not affect facets, illustration see below)
grouped_results = {
    "facet1_1+facet2_1+label_1": [
        item1_result_1, item1_result_2, ..., item1_result_16,  # 16 samples per item
        item2_result_1, item2_result_2, ..., item2_result_16,
        # ...
    ],
    "facet1_2+facet2_1+label_1": [
        item1_result_1, item1_result_2, ..., item1_result_16,
        # ...
    ],
    "facet1_1+facet2_2+label_1": [
        item1_result_1, item1_result_2, ..., item1_result_16,
        # ...
    ],
    "facet1_2+facet2_2+label_1": [
        item1_result_1, item1_result_2, ..., item1_result_16,
        # ...
    ],
    # ...
}
```

Metric code can group results further by item if needed (like in pass@k metrics).

#### Built-in Metric Interface
```python
def calculate_metric(
    evaluation_results: List[EvaluationResult], 
    facets: List[str],
    **params
) -> dict:
    """
    Built-in metrics expect:
    - evaluation_results[i]["item_id"] for grouping in facets combinations
    - evaluation_results[i]["label"] for use as one facet combined with other facets
    - evaluation_results[i]["passed"] for pass@k
    - evaluation_results[i]["score"] for numeric metrics
    """
```

Output example see below.

## Complete Data Flow Example

### Configuration
```yaml
sample_params:
  num_samples: 3  # Generate 3 samples per dataset item
```

### Dataset
```json
{"id": "problem_1", "data": {"question": "What is 2+2?", "answer": "4"}}
{"id": "problem_2", "data": {"question": "What is 3+3?", "answer": "6"}}
```

### Phase 1 Preprocessing

Use prompt templates to generate actual prompts for the model. For example, a simple template might be:
```yaml
prompt_template: "Solve the following problem: {{question}}"
```

### Phase 1 Output (responses.jsonl)
```json
{"item_id": "problem_1", "sample_id": "problem_1_sample_0", "sample_index": 0, "total_samples": 3, "response": "2+2=4", ...}
{"item_id": "problem_1", "sample_id": "problem_1_sample_1", "sample_index": 1, "total_samples": 3, "response": "4", ...}
...
```

a real response line might look like (pretty-printed for clarity):
```json
{
    "item_id": "20250510155538代丽00_05_31",
    "sample_id": "20250510155538代丽00_05_31_sample_10",
    "sample_index": 10,
    "total_samples": 16,
    "model_name": "Qwen3-1.7B-nothink",
    "prompt": "[expanded prompt]",
    "response": "[model raw response]",
    "inference_time": 17.994346618652344,
    "timestamp": 1750440075.5954432,
    "metadata": {
        "model_id": "Qwen3-1.7B-nothink",
        "prompt_template": "[prompt template]",
        "sampling": {
            "max_tokens": 3000,
            "num_samples": 16
        },
        "finish_reason": "stop",
        "response_id": "chatcmpl-CKRGbIUGdL3KwioMyzeCZsICET3xUeAj"
    },
    "error": null,
    "tokens_per_second": 301.8170159271249,
    "total_tokens": 5431,
    "prompt_tokens": 2929,
    "completion_tokens": 2502
}
```

### Phase 2 Evaluation

```python
# Evaluation method checks if response contains correct answer
def exact_match_eval(response: str, ground_truth: dict, inference_result=None, **params):
    correct_answer = ground_truth["answer"]
    return {
        "label": {"name": "exact_match"},
        "result": {
            "passed": correct_answer in response, "score": 1.0 if correct_answer in response else 0.0
        }
    }
```

### Phase 2 Output (evaluation_results.jsonl)
```jsonl
{"item_id": "problem_1", "sample_id": "problem_1_sample_0", "sample_index": 0, "label": "exact_match", "passed": true, "score": 1.0, ...}
{"item_id": "problem_1", "sample_id": "problem_1_sample_1", "sample_index": 1, "label": "exact_match", "passed": true, "score": 1.0, ...}
{"item_id": "problem_1", "sample_id": "problem_1_sample_2", "sample_index": 2, "label": "exact_match", "passed": true, "score": 1.0, ...}
{"item_id": "problem_2", "sample_id": "problem_2_sample_0", "sample_index": 0, "label": "exact_match", "passed": true, "score": 1.0, ...}
{"item_id": "problem_2", "sample_id": "problem_2_sample_1", "sample_index": 1, "label": "exact_match", "passed": true, "score": 1.0, ...}
{"item_id": "problem_2", "sample_id": "problem_2_sample_2", "sample_index": 2, "label": "exact_match", "passed": false, "score": 0.0, ...}
```

## Final Results (Metrics)

```jsonl
{"metric_name": "pass@k", "facets": ["metadata.model_id", "prompt_template"], "metadata.model_id": "model_1", "prompt_template": "template_1", "label": "exact_match", "pass_at_k": 0.67, "average_sample_count": 3, "total_sample_count": 6, ...}
{"metric_name": "pass@k", "facets": ["metadata.model_id", "prompt_template"], "metadata.model_id": "model_1", "prompt_template": "template_2", "label": "exact_match", "pass_at_k": 0.83, "average_sample_count": 3, "total_sample_count": 6, ...}
```

## Key Design Principles

1. **Unique Identifiers**: Every level has unique IDs (`item_id`, `sample_id`)
2. **Traceability**: Each result can be traced back to its source
3. **Grouping**: `item_id` enables proper grouping for pass@k metrics
4. **Granularity**: `label` allows for more granular analysis for one item and `facets` enable metrics grouping by model, prompt, etc.
5. **Compatibility**: Built-in metrics work with standard field names
6. **Extensibility**: Custom fields can be added without breaking the flow
