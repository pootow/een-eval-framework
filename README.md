# een_eval: Comprehensive Language Model Evaluation Framework

A flexible, extensible framework for evaluating language models with support for custom evaluation methods, metrics, and multiple model types including OpenAI API and local llama.cpp models.

## Features

- **🔧 Client-First Design**: Easy-to-use constructor and configuration-based API
- **🔄 Multiple Model Types**: OpenAI API, llama.cpp with Docker integration  
- **📊 Flexible Evaluation**: Built-in and custom evaluation methods and metrics
- **⚙️ Configuration-Driven**: YAML configuration with programmatic overrides
- **🔄 Resume Capability**: Checkpoint and resume interrupted evaluations
- **🐳 Docker Integration**: Seamless llama.cpp model deployment
- **📝 Comprehensive Logging**: Detailed progress tracking and error reporting
- **🎯 Multiple Modes**: Inference-only, evaluation-only, or full pipeline

## Quick Start

### Installation

```bash
cd een_eval/
pip install -e .
```

### Basic Usage

```python
from een_eval import EvalWorkflow, Model
from een_eval.core.evaluation import EvaluationMethod
from een_eval.core.metrics import Metric

# Simple evaluation using constructor pattern
def custom_evaluation(response: str, ground_truth: dict, inference_result=None, **params):
    expected = ground_truth.get("answer", "")
    is_correct = response.strip().lower() == expected.strip().lower()
    
    return {
        "labels": [
            {
                "label": {"name": "exact_match"},
                "result": {
                    "passed": is_correct,
                    "score": 1.0 if is_correct else 0.0,
                }
            }
        ]
    }

def accuracy_metric(results: list, **params):
    if not results:
        return []
    total = len(results)
    correct = sum(1 for r in results if r.get("passed", False))
    return [{"metric_name": "accuracy", "value": correct / total}]

workflow = EvalWorkflow(
    dataset="data/qa.jsonl",
    sample_params={"temperature": 0.0, "max_tokens": 100},
    eval_prompt_template="Question: {{ question }}\nAnswer:",
    evaluation_methods=[
        EvaluationMethod.from_function("qa_eval", custom_evaluation)
    ],
    metrics=[
        Metric.from_function("accuracy", accuracy_metric)
    ],
    output_dir="results/basic_example",
    mode="inference"
)

# Add models
workflow.add_models([Model.from_name("gpt-3.5-turbo")])

results = workflow.run()
print(f"Evaluation completed: {results}")
```

### Configuration-based Usage

```python
from een_eval import EvalWorkflow

# Load from configuration file
workflow = EvalWorkflow.from_config("config.yaml")
results = workflow.run()
```

### CLI Usage

```bash
# Run evaluation with configuration file
een-eval run --config config.yaml --verbose

# Run evaluation only (after inference completed)
een-eval eval --config config.yaml
```

## Architecture

The framework is built with a modular architecture:

```
een_eval/
├── core/           # Core evaluation components
│   ├── models.py   # Model abstractions (OpenAI, llama.cpp)
│   ├── evaluation.py  # Evaluation methods
│   ├── metrics.py  # Metrics computation
│   └── dataset.py  # Dataset handling
├── workflow/       # Workflow orchestration
│   ├── workflow.py # Main workflow orchestrator
│   ├── config.py   # Configuration management
│   ├── inference.py # Inference engine
│   └── evaluation.py # Evaluation engine
└── utils/          # Utility components
    ├── io.py       # File I/O operations
    ├── docker.py   # Docker management for llama.cpp
    └── validation.py # Configuration validation
```

## Key Components

### 1. Models

Support for multiple model types:

```python
# OpenAI models
workflow.add_models([Model.from_name("gpt-4")])

# llama.cpp models with Docker
model_config = ModelConfig(
    name="Qwen3-1.7B",
    type="llama.cpp", 
    model_path="unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf",
    endpoint="http://localhost:28000/v1",
    docker={
        "models_volume": "F:\\ai-models\\LLMs-LMStudio",
        "host_port": 1236,
        "n_gpu_layers": 99,
        "context_size": 8000
    }
)
workflow.add_models([Model.from_config(model_config)])
```

### 2. Evaluation Methods

Custom evaluation methods following the actual interface:

```python
def evaluate_qa_params(response: str, ground_truth: dict, inference_result=None, **params):
    """
    Custom evaluation function.
    
    Args:
        response: Model's response string
        ground_truth: Expected data from dataset  
        inference_result: Full inference context (optional)
        **params: Additional parameters
        
    Returns:
        Dict with labels list containing evaluation results
    """
    expected = ground_truth.get("answer", "")
    is_correct = expected.lower() in response.lower()
    
    return {
        "labels": [
            {
                "label": {"name": "qa_param_evaluation"},
                "result": {
                    "passed": is_correct,
                    "score": 1.0 if is_correct else 0.0,
                    # Additional custom fields
                    "total_params": 1,
                    "correct_params": 1 if is_correct else 0
                }
            }
        ]
    }

# Use in workflow
evaluation_method = EvaluationMethod.from_function(
    name="qa_evaluation", 
    function=evaluate_qa_params,
    params={"custom_param": "value"}
)
```

### 3. Metrics

Custom metrics following the actual interface:

```python
def calculate_pass_at_k(results: list, **params) -> list:
    """
    Calculate pass@k metric.
    
    Args:
        results: List of evaluation results
        **params: Parameters including k value
        
    Returns:
        List of metric dictionaries
    """
    k = params.get("k", 1)
    
    if not results:
        return []
    
    # Group by item_id for pass@k calculation
    grouped_results = {}
    for result in results:
        item_id = result.get("item_id", "default")
        if item_id not in grouped_results:
            grouped_results[item_id] = []
        grouped_results[item_id].append(result)
    
    # Calculate pass@k
    pass_count = 0
    total_items = len(grouped_results)
    
    for item_id, item_results in grouped_results.items():
        sorted_results = sorted(item_results, key=lambda x: x.get("score", 0), reverse=True)
        top_k_results = sorted_results[:k]
        if any(r.get("passed", False) for r in top_k_results):
            pass_count += 1
    
    pass_rate = pass_count / total_items if total_items > 0 else 0.0
    
    return [
        {
            "metric_name": f"pass@{k}",
            "value": pass_rate,
            "passed_items": pass_count,
            "total_items": total_items
        }
    ]

# Use in workflow
metric = Metric.from_function("pass_at_k", calculate_pass_at_k, params={"k": 1})
```

### 4. Configuration

YAML configuration matching the working prod_qa_params example:

```yaml
# Configuration file for evaluation
dataset:
  type: "custom"
  path: "eval.py"
  function_name: "load_transcript_dataset"
  params:
    save: true

models:
  - name: 'Qwen3-1.7B'
    type: "llama.cpp"
    model_path: "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf"
    endpoint: "http://localhost:28000/v1"
    docker:
      models_volume: "F:\\ai-models\\LLMs-LMStudio"
      host_port: 1236
      n_gpu_layers: 99
      context_size: 8000
    inference:
      think: false

# Custom evaluation method using function defined in eval.py
evaluation_methods:
  - name: "qc_rule_evaluation"
    type: "custom"
    path: "eval.py"
    function_name: "evaluate_qa_params"
    params:
      output_format: "json"

# Metrics for pass@1 and pass@16 evaluation
metrics:
  - name: "pass@1[mean@16]"
    type: "pass_at_k"
    facets:
      - model_name
    params:
      k: 1
      num_trials: 16

  - name: "pass@16"
    type: "pass_at_k"
    facets:
      - model_name
    params:
      k: 16

# Prompt template using file
prompt_template:
  path: "eval_prompt_template.md"

# Global sampling parameters
sample_params:
  max_tokens: 2000
  num_samples: 16  # Generate 16 samples for pass@16 evaluation

# Output configuration
output_dir: "output/evaluation_results/"

# Workflow settings
workflow:
  mode: "inference"  # Start with inference mode
  batch_size: 1
  max_workers: 1

resume: true

# Logging configuration
logging:
  level: DEBUG
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_config.yaml` - Simple evaluation setup
- `vllm_config.yaml` - Local model evaluation with llama.cpp
- `usage_examples.py` - Programmatic usage examples  
- `comprehensive_usage.py` - Advanced patterns and custom functions

Based on the working prod_qa_params implementation.

## Dataset Format

The framework supports JSONL, JSON, and CSV formats:

```jsonl
{"question": "What is 2+2?", "answer": "4"}
{"question": "Capital of France?", "answer": "Paris"}
```

## Output Structure

```
results/
├── inference.jsonl           # Model responses with metadata
├── responses.jsonl          # Sample-level detailed data
├── evaluation_results.json # Evaluation outcomes  
├── status.json             # Workflow status for resuming
└── config.yaml            # Used configuration
```

## Advanced Features

### Resume Capability

```python
# Enable resume for long-running evaluations
workflow = EvalWorkflow(
    dataset="large_dataset.jsonl",
    resume=True,
    output_dir="results/resumable"
)

# Automatically resumes from last checkpoint
results = workflow.run()
```

### Custom Functions Integration

Following the prod_qa_params pattern:

```python
# Custom dataset loader
def load_transcript_dataset(save: bool = False) -> List[Dict]:
    """Load custom dataset with preprocessing."""
    # Implementation details
    return dataset

# Custom evaluation with multiple criteria  
def evaluate_qa_params(response: str, ground_truth: dict, inference_result=None, **params) -> dict:
    """Evaluate response against multiple QA parameters."""
    # Multiple evaluation labels
    return {
        "labels": [
            {"label": {"name": "param1"}, "result": {"passed": True, "score": 1.0}},
            {"label": {"name": "param2"}, "result": {"passed": False, "score": 0.0}},
        ]
    }

# Usage in workflow
workflow = EvalWorkflow(
    dataset="eval.py:load_transcript_dataset",  # Custom loader
    evaluation_methods=[
        EvaluationMethod.from_function("qa_eval", evaluate_qa_params)
    ]
)
```

## Environment Setup

```bash
# API Keys for OpenAI models
export OPENAI_API_KEY="your-key"

# For llama.cpp with Docker (optional)
docker pull your-llama-cpp-image:latest
```

## CLI Reference

```bash
# Run evaluation with configuration file
een-eval run --config config.yaml --verbose

# Run evaluation only (after inference completed)
een-eval eval --config config.yaml --output-dir results/eval_20231201_120000

# Run full evaluation (inference + evaluation)
een-eval run --config config.yaml --mode full

# Check help
een-eval --help
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code quality checks pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
