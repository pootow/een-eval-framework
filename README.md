# een_eval: Comprehensive Language Model Evaluation Framework

A flexible, extensible framework for evaluating language models with support for custom evaluation methods, metrics, and multiple model types including OpenAI API and local VLLM models.

## Features

- **🔧 Client-First Design**: Easy-to-use API with builder pattern support
- **🔄 Multiple Model Types**: OpenAI API, VLLM with Docker integration
- **📊 Flexible Evaluation**: Built-in and custom evaluation methods and metrics
- **🎯 Pass@K Support**: Built-in support for pass@k metrics with proper sample coordination
- **⚙️ Configuration-Driven**: YAML/JSON configuration with programmatic overrides
- **🔄 Resume Capability**: Checkpoint and resume interrupted evaluations
- **🐳 Docker Integration**: Seamless VLLM model deployment
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
from een_eval import EvalWorkflow

# Simple evaluation
workflow = (EvalWorkflow.builder()
            .dataset_from_file("data/qa.jsonl", 
                             input_field="question",
                             expected_output_field="answer")
            .add_openai_model("gpt-3.5-turbo")
            .add_evaluation_method("exact_match")
            .add_metric("pass_rate")
            .build())

results = workflow.run()
print(f"Pass rate: {results['metrics']['pass_rate']}")
```

### CLI Usage

```bash
# Create sample configuration
een_eval create-config --output config.yaml --type basic

# Run evaluation
een_eval run --config config.yaml

# List available components
een_eval list-components
```

## Architecture

The framework is built with a modular architecture:

```
een_eval/
├── core/           # Core evaluation components (no I/O dependencies)
│   ├── models.py   # Model abstractions (OpenAI, VLLM)
│   ├── evaluation.py  # Evaluation methods
│   ├── metrics.py  # Metrics computation
│   └── dataset.py  # Dataset handling
├── workflow/       # Workflow orchestration (handles I/O)
│   ├── workflow.py # Main workflow orchestrator
│   ├── config.py   # Configuration management
│   ├── inference.py # Inference engine
│   └── evaluation.py # Evaluation engine
└── utils/          # Utility components
    ├── io.py       # File I/O operations
    ├── docker.py   # Docker management for VLLM
    └── validation.py # Configuration validation
```

## Key Components

### 1. Models

Support for multiple model types:

```python
# OpenAI models
model = OpenAIModel("gpt-4", temperature=0.0)

# VLLM models with Docker
model = VLLMModel(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    docker_image="vllm/vllm-openai:latest",
    gpu_count=1
)
```

### 2. Evaluation Methods

Built-in evaluation methods:
- `exact_match`: Exact string matching
- `contains`: Substring matching  
- `regex_match`: Regular expression patterns
- `json_match`: JSON structure validation
- `numeric_match`: Numeric comparisons
- `length_check`: Text length validation

Custom evaluation methods:
```python
def custom_evaluator(prediction: str, expected: str, threshold: float = 0.8) -> bool:
    # Custom evaluation logic
    return similarity_score > threshold

workflow.add_custom_evaluation_method("similarity", custom_evaluator)
```

### 3. Metrics

Built-in metrics:
- `pass_rate`: Overall pass percentage
- `pass_at_k`: Pass@K for code generation
- `mean`, `median`, `std`: Statistical measures
- `percentile`: Percentile distributions
- `count`, `min`, `max`, `sum`: Basic aggregations

### 4. Configuration

YAML/JSON configuration with full programmatic override capability:

```yaml
dataset:
  file_path: "data/eval.jsonl"
  input_field: "prompt"
  expected_output_field: "expected"

models:
  - name: "gpt-4"
    type: "openai"
    config:
      temperature: 0.0
      max_tokens: 512

evaluation_methods:
  - name: "exact_match"
    type: "exact_match"

metrics:
  - name: "pass_rate"
    type: "pass_rate"
```

## Advanced Features

### Resume Capability

```python
# Enable resume for long-running evaluations
workflow = (EvalWorkflow.builder()
            .dataset_from_file("large_dataset.jsonl")
            .enable_resume()
            .build())

# Automatically resumes from last checkpoint
results = workflow.run()
```

### Docker Integration for VLLM

```python
# Automatic Docker container management
workflow.add_vllm_model(
    "llama2-7b",
    model_path="meta-llama/Llama-2-7b-chat-hf",
    docker_image="vllm/vllm-openai:latest",
    gpu_count=1,
    port=8000
)
```

### Parallel Processing

```python
workflow = (EvalWorkflow.builder()
            .inference_config(batch_size=8, max_parallel=4)
            .build())
```

### Custom Functions

```python
# Custom evaluation method
def code_execution_check(prediction: str, test_cases: list) -> bool:
    # Execute code and run test cases
    return all_tests_pass

# Custom metric
def weighted_average(scores: list, weights: list) -> float:
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

workflow = (EvalWorkflow.builder()
            .add_custom_evaluation_method("code_exec", code_execution_check)
            .add_custom_metric("weighted_avg", weighted_average)
            .build())
```

### Pass@K Evaluation Support

The framework provides built-in support for pass@k metrics, which are essential for evaluating code generation and other tasks where multiple attempts may be needed to find a correct solution.

#### Configuration

```yaml
# Sample parameters - specify num_samples for pass@k evaluation
sample_params:
  temperature: 0.7
  max_tokens: 1024
  num_samples: 16  # Generate 16 samples per dataset item

# Built-in pass@k metrics
metrics:
  - name: "pass_at_1"
    type: "built_in"
    params:
      k: 1
  - name: "pass_at_16"
    type: "built_in"
    params:
      k: 16
```

#### How It Works

1. **Inference Phase**: For each dataset item, the framework generates `num_samples` responses
2. **Sample Tracking**: Each response is tagged with `item_id`, `sample_id`, and `sample_index`
3. **Evaluation Phase**: Evaluation methods are applied to each sample individually
4. **Metric Calculation**: Pass@k metrics group samples by `item_id` and calculate the probability that at least one sample passes

#### Example Usage

```python
workflow = (EvalWorkflow.builder()
    .dataset_from_file("problems.jsonl")
    .add_vllm_model("codellama-7b")
    .sample_params(temperature=0.8, num_samples=16)
    .add_evaluation_method("code_execution")
    .add_metric("pass_at_1")
    .add_metric("pass_at_16")
    .build())

results = workflow.run()
print(f"Pass@1: {results['metrics']['pass_at_1']:.2%}")
print(f"Pass@16: {results['metrics']['pass_at_16']:.2%}")
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_config.yaml` - Simple OpenAI evaluation setup
- `vllm_config.yaml` - Local model evaluation with VLLM
- `usage_examples.py` - Programmatic usage examples

## Dataset Format

The framework supports JSONL, JSON, and CSV formats:

```jsonl
{"prompt": "What is 2+2?", "expected": "4"}
{"prompt": "Capital of France?", "expected": "Paris"}
```

## Output Structure

```
results/
├── predictions/           # Model predictions
├── evaluations/          # Evaluation results
├── metrics/             # Computed metrics
├── intermediate/        # Checkpoints (if enabled)
├── status.json         # Current status
└── config.yaml        # Used configuration
```

## Environment Setup

```bash
# API Keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Docker for VLLM (optional)
docker pull vllm/vllm-openai:latest
```

## CLI Reference

Full CLI documentation available in `CLI_DOCUMENTATION.md`.

Quick reference:
```bash
een_eval create-config --output config.yaml
een_eval validate --config config.yaml  
een_eval run --config config.yaml
een_eval resume --resume-dir results/eval_xyz
een_eval list-components
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
black een_eval/
flake8 een_eval/
mypy een_eval/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code quality checks pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: See `examples/` and `CLI_DOCUMENTATION.md`
- Issues: GitHub Issues
- Examples: `examples/usage_examples.py`

---

**een_eval** - Comprehensive, flexible, and extensible language model evaluation made simple.
