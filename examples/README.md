# een_eval Examples

This directory contains example configurations and usage scripts for the `een_eval` framework.

## Files

### Configuration Examples

- **`basic_config.yaml`** - Simple evaluation setup using OpenAI models
- **`vllm_config.yaml`** - Advanced setup using local VLLM models with Docker

### Usage Scripts

- **`usage_examples.py`** - Comprehensive examples showing programmatic usage

## Quick Start

### 1. Using Configuration Files

```bash
# Create a sample configuration
een_eval create-config --output my_config.yaml --type basic

# Validate the configuration
een_eval validate --config my_config.yaml

# Run evaluation
een_eval run --config my_config.yaml
```

### 2. Using Command Line (Minimal Setup)

```bash
# Run with minimal command line arguments
een_eval run \
  --dataset data/eval_dataset.jsonl \
  --input-field prompt \
  --expected-field expected \
  --model-type openai \
  --model-name gpt-3.5-turbo \
  --output-dir results
```

### 3. Programmatic Usage

```python
from een_eval import EvalWorkflow

# Simple evaluation
workflow = EvalWorkflow(
    dataset="data/qa.jsonl",
    sample_params={"temperature": 0.0, "max_tokens": 100}
).add_evaluation_method("exact_match") \
 .add_metric("pass_rate")

results = workflow.run()
```

## Configuration Options

### Dataset Configuration

```yaml
dataset:
  file_path: "path/to/dataset.jsonl"
  input_field: "prompt"
  expected_output_field: "expected"
  sample_size: 100  # Optional: limit dataset size
  filter_condition: 'difficulty == "medium"'  # Optional: filter data
```

### Model Configuration

#### OpenAI Models

```yaml
models:
  - name: "gpt-4"
    type: "openai"
    config:
      model: "gpt-4"
      temperature: 0.0
      max_tokens: 512
```

#### VLLM Models

```yaml
models:
  - name: "llama2"
    type: "vllm"
    config:
      model: "meta-llama/Llama-2-7b-chat-hf"
      docker_image: "vllm/vllm-openai:latest"
      gpu_count: 1
      port: 8000
```

### Evaluation Methods

```yaml
evaluation_methods:
  - name: "exact_match"
    type: "exact_match"
  
  - name: "contains_check"
    type: "contains"
    config:
      case_sensitive: false
  
  - name: "custom_eval"
    type: "custom"
    function: "my_module.my_function"
```

### Metrics

```yaml
metrics:
  - name: "pass_rate"
    type: "pass_rate"
  
  - name: "pass_at_k"
    type: "pass_at_k"
    config:
      k: 5
  
  - name: "custom_metric"
    type: "custom"
    function: "my_module.my_metric"
```

## Environment Setup

### API Keys

Set environment variables for model access:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # If using Anthropic models
```

### Docker for VLLM

Make sure Docker is installed and running for VLLM models:

```bash
# Pull VLLM image
docker pull vllm/vllm-openai:latest

# Check GPU availability
nvidia-smi
```

## Common Use Cases

### 1. Code Evaluation

```yaml
evaluation_methods:
  - name: "syntax_check"
    type: "regex_match"
    config:
      pattern: "def\\s+\\w+\\s*\\("
  
  - name: "execution_test"
    type: "custom"
    function: "code_evaluators.execute_and_test"
```

### 2. JSON Response Validation

```yaml
evaluation_methods:
  - name: "json_structure"
    type: "json_match"
    config:
      required_keys: ["answer", "confidence"]
      strict: false
```

### 3. Multi-Model Comparison

```yaml
models:
  - name: "gpt-4"
    type: "openai"
    config: { model: "gpt-4" }
  
  - name: "gpt-3.5"
    type: "openai" 
    config: { model: "gpt-3.5-turbo" }
  
  - name: "llama2"
    type: "vllm"
    config: { model: "meta-llama/Llama-2-7b-chat-hf" }
```

## Resume Functionality

If an evaluation is interrupted, you can resume it:

```bash
# Resume from the last checkpoint
een_eval resume --resume-dir results/eval_20231201_120000
```

The framework automatically saves progress and can continue from where it left off.

## Tips

1. **Start Small**: Test with a small sample size first
2. **Use Validation**: Always validate configurations before running
3. **Monitor Progress**: Use `--verbose` flag for detailed progress information
4. **Save Intermediate Results**: Enable `save_intermediate` for long-running evaluations
5. **Custom Functions**: Create custom evaluation methods and metrics for domain-specific needs
