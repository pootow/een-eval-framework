# een_eval CLI Documentation

The `een_eval` framework provides a comprehensive command-line interface for running language model evaluations.

## Installation

```bash
cd e:\yanji\qcpoint-model
pip install -e een_eval/
```

## Basic Usage

### 1. List Available Components

```bash
python -m een_eval.cli list-components
```

This command shows all built-in evaluation methods and metrics available in the framework.

### 2. Create Sample Configuration

```bash
# Create basic configuration
python -m een_eval.cli create-config --output config.yaml --type basic

# Create VLLM configuration
python -m een_eval.cli create-config --output vllm_config.yaml --type vllm
```

### 3. Validate Configuration

```bash
python -m een_eval.cli validate --config config.yaml
```

### 4. Run Evaluation

```bash
# Full evaluation (inference + evaluation)
python -m een_eval.cli run --config config.yaml --mode full

# Inference only, mode is default to 'inference'
python -m een_eval.cli run --config config.yaml

# Evaluation only (requires existing predictions)
python -m een_eval.cli run --config config.yaml --mode evaluation --output-dir results/eval_20231201_120000

# Or you can use the short form
python -m een_eval.cli eval --config config.yaml --output-dir results/eval_20231201_120000
```

### 5. Resume Interrupted Evaluation

```bash
python -m een_eval.cli resume --output-dir results/eval_20231201_120000 --config config.yaml
```

This will only resume the inference mode.

## Command Line Options

### Global Options

- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging level
- `--log-file FILE`: Log to file in addition to console
- `--verbose, -v`: Enable verbose output

### Run Command Options

- `--config, -c FILE`: Configuration file path
- `--mode {inference,evaluation,full}`: Evaluation mode (default: full)
- `--output-dir, -o DIR`: Output directory (overrides config)
- `--batch-size INT`: Inference batch size (overrides config)
- `--max-workers INT`: Maximum parallel processes (overrides config)

#### Quick Run Options (without config file)

- `--dataset FILE`: Dataset file path
- `--input-field STR`: Input field name in dataset
- `--expected-field STR`: Expected output field name in dataset
- `--model-type {openai,vllm}`: Model type
- `--model-name STR`: Model name/identifier
- `--temperature FLOAT`: Model temperature
- `--max-tokens INT`: Maximum tokens for model generation

## Examples

### Example 1: Quick Evaluation with OpenAI

```bash
python -m een_eval.cli run \
  --dataset data/qa_dataset.jsonl \
  --input-field question \
  --expected-field answer \
  --model-type openai \
  --model-name gpt-3.5-turbo \
  --output-dir results/quick_eval
```

### Example 2: Configuration-based Evaluation

```bash
# Create config
python -m een_eval.cli create-config --output eval_config.yaml --type basic

# Edit the config file as needed
# ...

# Validate config
python -m een_eval.cli validate --config eval_config.yaml

# Run evaluation
python -m een_eval.cli run --config eval_config.yaml --verbose
```

### Example 3: Resume Interrupted Evaluation

```bash
# If evaluation is interrupted, resume from checkpoint
python -m een_eval.cli resume --output-dir results/eval_20231201_120000 --verbose
```

## Configuration File Format

The framework supports both YAML and JSON configuration files. Here's a basic example:

```yaml
dataset:
  file_path: "data/eval_dataset.jsonl"
  input_field: "prompt"
  expected_output_field: "expected_response"

models:
  - name: "gpt-4"
    type: "openai"
    config:
      model: "gpt-4"
      temperature: 0.0
      max_tokens: 512

evaluation_methods:
  - name: "exact_match"
    type: "exact_match"

metrics:
  - name: "pass_rate"
    type: "pass_rate"

output:
  directory: "results"
  save_predictions: true
  save_evaluations: true
```

## Environment Variables

Set these environment variables for API access:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic models)

## Output Structure

The framework creates the following output structure:

```
results/
├── predictions/
│   ├── model1_predictions.jsonl
│   └── model2_predictions.jsonl
├── evaluations/
│   ├── model1_evaluations.jsonl
│   └── model2_evaluations.jsonl
├── metrics/
│   └── summary_metrics.json
├── status.json
└── config.yaml
```

## Built-in Components

### Evaluation Methods

- `exact_match`: Exact string matching
- `contains`: Substring matching
- `regex_match`: Regular expression matching
- `json_match`: JSON structure validation
- `numeric_match`: Numeric value comparison
- `length_check`: Text length validation

### Metrics

- `pass_rate`: Overall pass rate
- `pass_at_k`: Pass at K metric
- `mean`: Mean score
- `median`: Median score
- `percentile`: Percentile scores
- `count`: Count of samples
- `std`: Standard deviation
- `min`: Minimum score
- `max`: Maximum score
- `sum`: Sum of scores

## Error Handling

The CLI provides helpful error messages and exit codes:

- `0`: Success
- `1`: General error
- `130`: Interrupted by user (Ctrl+C)

Use `--verbose` flag for detailed error information and stack traces.
