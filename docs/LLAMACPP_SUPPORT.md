# Llama.cpp Model Support

The een_eval framework now supports running models using llama.cpp via Docker containers. This enables running GGUF quantized models with efficient GPU acceleration.

## Configuration

To use a llama.cpp model, set `type: "llama.cpp"` in your model configuration:

```yaml
models:
  - name: 'Qwen3-1.7B'
    type: "llama.cpp"
    params:
      model_path: "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf"
```

## Docker Command Generated

The framework will automatically start a Docker container using the command equivalent to:

```bash
docker run -d --rm \
  --gpus all \
  --ipc=host \
  -p 1234:8000 \
  -v /home/pootow/LLMs/:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf \
  --port 8000 \
  --host 0.0.0.0 \
  -c 8000 \
  --n-gpu-layers 99 \
  --no-mmap
```

## Configuration Options

### Required Parameters
- `name`: Display name for the model
- `type`: Must be `"llama.cpp"`
- `params.model_path`: Path to the GGUF model file (relative to the models volume)

### Optional Parameters
- `endpoint`: OpenAI-compatible API endpoint (defaults to `http://localhost:1234/v1`)
- `docker_image`: Docker image to use (defaults to `ghcr.io/ggml-org/llama.cpp:server-cuda`)
- `docker_params`: Additional Docker parameters

### Docker Parameters (in `docker_params`)
- `models_volume`: Host path to mount as /models (default: `/home/pootow/LLMs/`)
- `context_size`: Context size in tokens (default: 8000, corresponds to `-c` parameter)
- `n_gpu_layers`: Number of layers to run on GPU (default: 99, corresponds to `--n-gpu-layers`)
- `no_mmap`: Disable memory mapping (default: true, adds `--no-mmap` flag)
- `host_port`: Host port to bind to (default: 1234)
- `container_port`: Container port (default: 8000)
- `host`: Host to bind to (default: "0.0.0.0")

## Complete Configuration Example

```yaml
models:
  - name: 'Qwen3-1.7B'
    type: "llama.cpp"
    endpoint: "http://localhost:1234/v1"
    params:
      model_path: "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf"
    docker_params:
      models_volume: "/path/to/your/models"
      context_size: 4096
      n_gpu_layers: 50
      host_port: 8080
```

## Usage in Code

```python
from een_eval.core.models import Model, ModelConfig, ModelType

# Create from dictionary (like YAML config)
config_dict = {
    "name": "Qwen3-1.7B",
    "type": "llama.cpp",
    "params": {
        "model_path": "unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q4_K_XL.gguf"
    }
}

model_config = ModelConfig.from_dict(config_dict)
model = Model.from_config(model_config)

# Use the model
with model:
    result = model.generate("Hello, how are you?")
    print(result.response)
```

## Requirements

1. Docker with NVIDIA runtime support
2. NVIDIA GPU with CUDA support
3. The specified GGUF model file in the models volume
4. Sufficient GPU memory for the model

## Notes

- The container runs with `--rm` flag, so it will be automatically removed when stopped
- The framework automatically waits for the container to be ready before proceeding
- Uses OpenAI-compatible API for inference, so all standard parameters (temperature, top_p, etc.) work
- Container logs can be accessed through Docker commands if troubleshooting is needed
