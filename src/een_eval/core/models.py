"""
Model abstractions for the evaluation framework.

This module provides model configuration and interface abstractions
that support both local VLLM models and OpenAI API models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import openai
import time
import random
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    OPENAI = "openai"
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"
    MOCK = "mock"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str # display name for the model, used in UI and logs
    type: ModelType
    
    # Model-specific parameters (not Docker-related)
    model_path: Optional[str] = None
    _model_name: Optional[str] = None  # For OpenAI-compatible models, this is the model name, if not provided, defaults to name.
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def model_name(self) -> str:
        """Return the model name, defaulting to the config name if not set."""
        return self._model_name or self.name
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value

    # Docker configuration (only for local models)
    docker_config: Optional[Dict[str, Any]] = None
    
    # Inference parameters
    inference_params: Optional[Dict[str, Any]] = None
    
    # Connection parameters
    timeout: int = 300
    max_retries: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        model_type = ModelType(data.get("type", "openai"))

        # construct a endpoint if not provided
        endpoint = data.get("endpoint")
        if not endpoint:
            port = None
            if data.get("docker") and data["docker"].get("host_port"):
                port = data["docker"]["host_port"]
            endpoint = f"http://localhost:{port if port else 1234}/v1"
        
        return cls(
            name=data["name"],
            type=model_type,
            model_path=data.get("model_path"),
            _model_name=data.get("model_name"),
            endpoint=endpoint,
            api_key=data.get("api_key"),
            docker_config=data.get("docker", {}),
            inference_params=data.get("inference", {}),
            timeout=data.get("timeout", 300),
            max_retries=data.get("max_retries", 3)
        )
    
    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """Create ModelConfig from model name (defaults to local OpenAI API compatible service at localhost:1234)."""
        return cls(
            name=name, 
            type=ModelType.OPENAI,
            api_key="dummy_api_key_for_local_model",
            endpoint="http://localhost:1234/v1"
        )

    def merge_inference_params(self, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Merge global sample_params, model inference_params, and kwargs.
        
        Priority: kwargs > model.inference_params > global_sample_params
        
        Args:
            global_sample_params: Global sampling parameters from config
            **kwargs: Call-specific overrides
            
        Returns:
            Merged parameters dictionary
        """
        result = {}
        
        # Start with global parameters (lowest priority)
        if global_sample_params:
            result.update(global_sample_params)
        
        # Override with model-specific parameters
        if self.inference_params:
            result.update(self.inference_params)
        
        # Override with call-specific parameters (highest priority)
        result.update(kwargs)
        
        return result


@dataclass
class SimpleInferenceResult:
    """Simple result from model inference (before workflow processing)."""
    response: str
    prompt: str
    inference_time: float
    model_name: str
    
    # Optional fields
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """Result from model inference following DataFlow.md specification."""
    item_id: str                         # Links back to original dataset item
    sample_id: str                       # Unique per sample: "{item_id}_sample_{index}"
    sample_index: int                    # 0 to num_samples-1
    total_samples: int                   # Total samples (= model count * items count * num_samples)
    model_name: str                      # Model name (just a friendly display name)
    prompt: str                          # Processed prompt sent to model
    response: str                        # Model's raw response
    inference_time: float                # Time taken for this inference
    timestamp: float                     # When this inference was made
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional inference-specific metadata
    error: Optional[str] = None          # Error message if inference failed
    
    # Additional fields for compatibility
    tokens_per_second: Optional[float] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ModelInterface(ABC):
    """Abstract interface for model inference."""

    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        global_sample_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SimpleInferenceResult:
        """Generate response for a prompt.
        
        Args:
            prompt: The input prompt
            global_sample_params: Global sampling parameters from config
            **kwargs: Override parameters for this specific generation
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available for inference."""
        pass
    
    @abstractmethod
    def startup(self) -> None:
        """Start up the model if needed."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the model if needed."""
        pass


class OpenAIModel(ModelInterface):
    """OpenAI API model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.endpoint
        )

    def _openai_chat_completion(self, params: dict):
        """Non-streaming OpenAI chat completion call."""
        response = self.client.chat.completions.create(**params)
        return response

    def _openai_chat_completion_stream(self, params: dict):
        """Streaming OpenAI chat completion call."""
        return self.client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True})

    def _openai_chat_completion_stream_to_string(self, params: dict):
        """Use streaming OpenAI API to simulate non-streaming response (aggregate chunks)."""
        stream = self._openai_chat_completion_stream(params)
        content = ""
        reasoning_content = ""
        response_id = None
        finish_reason = None
        usage = None
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                if hasattr(chunk.choices[0], 'finish_reason'):
                    finish_reason = chunk.choices[0].finish_reason
            if hasattr(chunk, 'id'):
                response_id = chunk.id
            if hasattr(chunk, 'usage'):
                usage = chunk.usage
        # Compose a mock response object similar to non-streaming
        class MockResponse:
            def __init__(self, content, reasoning_content, finish_reason, response_id, usage):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content, 'reasoning_content': reasoning_content}), 'finish_reason': finish_reason})]
                self.id = response_id
                self.usage = usage
        return MockResponse(content, reasoning_content, finish_reason, response_id, usage)

    def generate(self, prompt: str, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> SimpleInferenceResult:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        # Merge parameters using the config method
        merged_params = self.config.merge_inference_params(global_sample_params, **kwargs)

        # Build API parameters
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": merged_params.get("max_tokens", -1),
        }
        # Only add temperature/top_p if present in merged_params
        if "temperature" in merged_params:
            params["temperature"] = merged_params["temperature"]
        if "top_p" in merged_params:
            params["top_p"] = merged_params["top_p"]
        
        # Add any additional parameters from merged_params
        for key, value in merged_params.items():
            if key not in params and key not in ['num_samples', 'think']:  # Skip workflow-level params
                params[key] = value
        
        # Remove None values and handle max_tokens
        if params.get("max_tokens", -1) <= 0:
            params.pop("max_tokens", None)
        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = self._openai_chat_completion_stream_to_string(params)
            reasoning_content = ""
            if response.choices[0].message.reasoning_content:
                # If reasoning content is present, use it instead of the main content
                reasoning_content = f"<think>{response.choices[0].message.reasoning_content}</think>\n"
            content = reasoning_content + response.choices[0].message.content

            end_time = time.time()
            inference_time = end_time - start_time
            
            # Calculate tokens per second if usage info is available
            total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
            tokens_per_second = total_tokens / inference_time if total_tokens else 0
            
            return SimpleInferenceResult(
                response=content,
                prompt=prompt,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                model_name=self.config.name,
                total_tokens=total_tokens,
                prompt_tokens=getattr(response.usage, 'prompt_tokens', None) if response.usage else None,
                completion_tokens=getattr(response.usage, 'completion_tokens', None) if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return SimpleInferenceResult(
                response="",
                prompt=prompt,
                inference_time=time.time() - start_time,
                model_name=self.config.name,
                error=f"OpenAI API call failed: {e}"
            )
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            self.client.models.list()
            return True
        except:
            return False
    
    def startup(self) -> None:
        """No startup needed for OpenAI API."""
        pass
    
    def shutdown(self) -> None:
        """No shutdown needed for OpenAI API."""
        pass


class VLLMModel(ModelInterface):
    """VLLM model implementation (via OpenAI-compatible API)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.docker_container_id: Optional[str] = None
        self.client: Optional[openai.OpenAI] = None

    def generate(self, prompt: str, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> SimpleInferenceResult:
        """Generate response using VLLM API."""
        if not self.client:
            return SimpleInferenceResult(
                response="",
                prompt=prompt,
                inference_time=0.0,
                model_name=self.config.name,
                error="VLLM model not started. Call startup() first."
            )
        
        start_time = time.time()

        # Merge parameters using the config method
        merged_params = self.config.merge_inference_params(global_sample_params, **kwargs)
        
        # Build API parameters
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": merged_params.get("max_tokens", -1),
        }
        if "temperature" in merged_params:
            params["temperature"] = merged_params["temperature"]
        if "top_p" in merged_params:
            params["top_p"] = merged_params["top_p"]
        
        # Add any additional parameters from merged_params
        for key, value in merged_params.items():
            if key not in params and key not in ['num_samples', 'think']:  # Skip workflow-level params
                params[key] = value
        
        # Remove None values and handle max_tokens
        if params.get("max_tokens", -1) <= 0:
            params.pop("max_tokens", None)
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**params)
            end_time = time.time()
            
            inference_time = end_time - start_time
            reasoning_content = ""
            if response.choices[0].message.reasoning_content:
                # If reasoning content is present, use it instead of the main content
                reasoning_content = f"<think>{response.choices[0].message.reasoning_content}</think>\n"
            content = reasoning_content + response.choices[0].message.content
            
            # Calculate tokens per second if usage info is available
            total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
            tokens_per_second = total_tokens / inference_time if total_tokens else 0
            
            return SimpleInferenceResult(
                response=content,
                prompt=prompt,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                model_name=self.config.name,
                total_tokens=total_tokens,
                prompt_tokens=getattr(response.usage, 'prompt_tokens', None) if response.usage else None,
                completion_tokens=getattr(response.usage, 'completion_tokens', None) if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
        except Exception as e:
            logger.error(f"VLLM API call failed: {e}")
            return SimpleInferenceResult(
                response="",
                prompt=prompt,
                inference_time=time.time() - start_time,
                model_name=self.config.name,
                error=f"VLLM API call failed: {e}"
            )

    def is_available(self) -> bool:
        """Check if VLLM server is available by testing completion API."""
        if not self.config.endpoint:
            return False
        
        try:
            # Test with a simple completion request to ensure the API is actually working
            import requests
            test_payload = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": "/nothink\necho this word, and do not output anything else: model-ready."}],
                "max_tokens": 20,
                "temperature": 0.0
            }
            response = requests.post(
                f"{self.config.endpoint}/chat/completions",
                json=test_payload,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def startup(self) -> None:
        """Start VLLM server in Docker container."""
        from ..utils.docker import DockerManager
        
        if self.is_available():
            self.client = openai.OpenAI(base_url=self.config.endpoint, api_key="DUMMY_API_KEY")
            return
        
        docker_manager = DockerManager()
        
        # Default Docker parameters
        default_params = {
            "gpu_memory_utilization": 0.3,
            "swap_space": 1,
            "max_model_len": 8000,
            "enable_reasoning": True,
            "reasoning_parser": "deepseek_r1"
        }

        docker_params = {**default_params}
        if self.config.docker_config:
            docker_params.update(self.config.docker_config)
        
        # Validate required parameters
        if not self.config.model_path:
            raise ValueError("model_path is required for VLLM models")
        
        # Get docker image with default
        docker_image = docker_params.get("image", "vllm/vllm-openai:latest")
        
        self.docker_container_id = docker_manager.start_vllm_container(
            model_path=self.config.model_path,
            model_name=self.config.model_name,
            image=docker_image,
            host_port=self._extract_port_from_endpoint(),
            **docker_params
        )
        
        # Wait for server to be ready
        self._wait_for_server()
        
        # Setup client
        self.client = openai.OpenAI(base_url=self.config.endpoint, api_key="DUMMY_API_KEY")
    
    def shutdown(self) -> None:
        """Shutdown VLLM Docker container."""
        if self.docker_container_id:
            from ..utils.docker import DockerManager
            docker_manager = DockerManager()
            docker_manager.stop_container(self.docker_container_id)
            self.docker_container_id = None
        
        self.client = None
    
    def _extract_port_from_endpoint(self) -> int:
        """Extract port from endpoint URL."""
        if not self.config.endpoint:
            return 8000
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.config.endpoint)
            return parsed.port or 8000
        except:
            return 8000
    
    def _wait_for_server(self, max_wait: int = 300) -> None:
        """Wait for VLLM server to be ready."""
        import time

        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.is_available():
                return
            time.sleep(5)
        
        raise RuntimeError(f"VLLM server failed to start within {max_wait} seconds")


class LlamaCppModel(ModelInterface):
    """Llama.cpp model implementation (via OpenAI-compatible API)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.docker_container_id: Optional[str] = None
        self.client: Optional[openai.OpenAI] = None

    def generate(self, prompt: str, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> SimpleInferenceResult:
        """Generate response using llama.cpp API."""
        if not self.client:
            return SimpleInferenceResult(
                response="",
                prompt=prompt,
                inference_time=0.0,
                model_name=self.config.name,
                error="Llama.cpp model not started. Call startup() first."
            )
        
        start_time = time.time()
        
        # Merge parameters using the config method
        merged_params = self.config.merge_inference_params(global_sample_params, **kwargs)
        
        # Build API parameters
        params = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": merged_params.get("max_tokens", -1),
        }
        if "temperature" in merged_params:
            params["temperature"] = merged_params["temperature"]
        if "top_p" in merged_params:
            params["top_p"] = merged_params["top_p"]
        
        # Add any additional parameters from merged_params
        for key, value in merged_params.items():
            if key not in params and key not in ['num_samples', 'think']:  # Skip workflow-level params
                params[key] = value
        
        # Remove None values and handle max_tokens
        if params.get("max_tokens", -1) <= 0:
            params.pop("max_tokens", None)
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**params)
            end_time = time.time()
            
            inference_time = end_time - start_time
            reasoning_content = ""
            if response.choices[0].message.reasoning_content:
                # If reasoning content is present, use it instead of the main content
                reasoning_content = f"<think>{response.choices[0].message.reasoning_content}</think>\n"
            content = reasoning_content + response.choices[0].message.content
            
            # Calculate tokens per second if usage info is available
            total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
            tokens_per_second = total_tokens / inference_time if total_tokens and inference_time > 0 else 0
            
            return SimpleInferenceResult(
                response=content,
                prompt=prompt,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                model_name=self.config.name,
                total_tokens=total_tokens,
                prompt_tokens=getattr(response.usage, 'prompt_tokens', None) if response.usage else None,
                completion_tokens=getattr(response.usage, 'completion_tokens', None) if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
        except Exception as e:
            logger.error(f"Llama.cpp API call failed: {e}")
            return SimpleInferenceResult(
                response="",
                prompt=prompt,
                inference_time=time.time() - start_time,
                model_name=self.config.name,
                error=f"Llama.cpp API call failed: {e}"
            )
    
    def is_available(self) -> bool:
        """Check if llama.cpp server is available by testing completion API."""
        if not self.config.endpoint:
            return False
        
        try:
            # Test with a simple completion request to ensure the API is actually working
            import requests
            test_payload = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": "/nothink\necho this word, and do not output anything else: model-ready."}],
                "max_tokens": 20,
                "temperature": 0.0
            }
            response = requests.post(
                f"{self.config.endpoint}/chat/completions",
                json=test_payload,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def startup(self) -> None:
        """Start llama.cpp server in Docker container."""
        from ..utils.docker import DockerManager
        
        if self.is_available():
            self.client = openai.OpenAI(base_url=self.config.endpoint, api_key="DUMMY_API_KEY")
            return
        
        # Check that models_volume is provided in docker config
        if not self.config.docker_config or not self.config.docker_config.get("models_volume"):
            raise ValueError("models_volume is required in docker config for llama.cpp models")

        docker_manager = DockerManager()
        
        # Default Docker parameters for llama.cpp
        default_params = {
            "context_size": 8000,
            "n_gpu_layers": 99,
            "no_mmap": True,
        }
        
        docker_params = {**default_params}
        if self.config.docker_config:
            docker_params.update(self.config.docker_config)

        # Validate required parameters
        if not self.config.model_path:
            raise ValueError("model_path is required for llama.cpp models")
        
        # Get Docker image with default
        docker_image = docker_params.get("image", "ghcr.io/ggml-org/llama.cpp:server-cuda")

        if "host_port" not in docker_params:
            docker_params["host_port"] = self._extract_port_from_endpoint()
        
        self.docker_container_id = docker_manager.start_llamacpp_container(
            model_path=self.config.model_path,
            model_name=self.config.model_name,
            image=docker_image,
            **docker_params
        )
        
        # Wait for server to be ready
        self._wait_for_server(**{key: v for key, v in docker_params.items() if key in ['max_wait']})
        
        # Setup client
        self.client = openai.OpenAI(base_url=self.config.endpoint, api_key="DUMMY_API_KEY")
    
    def shutdown(self) -> None:
        """Shutdown llama.cpp Docker container."""
        if self.docker_container_id:
            from ..utils.docker import DockerManager
            docker_manager = DockerManager()
            docker_manager.stop_container(self.docker_container_id)
            self.docker_container_id = None
        
        self.client = None
    
    def _extract_port_from_endpoint(self) -> int:
        """Extract port from endpoint URL."""
        if not self.config.endpoint:
            return 1234  # Default port for llama.cpp
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.config.endpoint)
            return parsed.port or 1234
        except:
            return 1234
    
    def _wait_for_server(self, max_wait: int = 300) -> None:
        """Wait for llama.cpp server to be ready."""
        import time

        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.is_available():
                return
            time.sleep(5)
        
        raise RuntimeError(f"Llama.cpp server failed to start within {max_wait} seconds")


class MockModel(ModelInterface):
    """Mock model implementation for testing purposes."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.call_count = 0
        # Use mock_responses from inference_params if available, otherwise use defaults
        if config.inference_params and 'mock_responses' in config.inference_params:
            self.responses = config.inference_params['mock_responses']
        else:
            self.responses = [
                "This is a mock response for testing purposes.",
                "Another mock response to simulate model behavior.",
                "Mock models are useful for testing the evaluation framework.",
                "Testing with mock models ensures reproducible results.",
                "The framework supports various model types including mocks."
            ]

    def generate(self, prompt: str, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> SimpleInferenceResult:
        """Generate mock response."""
        start_time = time.time()
        
        # Check for failure_rate configuration to randomly throw exceptions
        if self.config.inference_params:
            failure_rate = self.config.inference_params.get('failure_rate', 0.0)
            if random.random() < failure_rate:
                # Simulate different types of failures that might occur
                failure_types = [
                    "ConnectionError: Mock network failure",
                    "TimeoutError: Mock request timeout",
                    "RuntimeError: Mock internal server error",
                    "ValueError: Mock invalid parameters",
                    "APIError: Mock API rate limit exceeded"
                ]
                error_msg = random.choice(failure_types)
                logger.error(f"Mock model failure: {error_msg}")
                return SimpleInferenceResult(
                    response="",
                    prompt=prompt,
                    inference_time=time.time() - start_time,
                    model_name=self.config.name,
                    error=error_msg
                )
        
        # Cycle through predefined responses
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        # Simulate some processing time
        time.sleep(0.1)
        end_time = time.time()
        
        inference_time = end_time - start_time
        mock_tokens = len(response.split()) + len(prompt.split())
        tokens_per_second = mock_tokens / inference_time if inference_time > 0 else 0
        
        return SimpleInferenceResult(
            response=response,
            prompt=prompt,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            model_name=self.config.name,
            total_tokens=mock_tokens,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response.split()),
            metadata={"mock": True, "call_count": self.call_count}
        )
    
    def is_available(self) -> bool:
        """Mock model is always available."""
        return True
    
    def startup(self) -> None:
        """No startup needed for mock model."""
        pass
    
    def shutdown(self) -> None:
        """No shutdown needed for mock model."""
        pass


class Model:
    """High-level model wrapper."""
    
    def __init__(self, interface: ModelInterface):
        self.interface = interface
        self.config = interface.config
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> "Model":
        """Create model from configuration."""
        if config.type == ModelType.OPENAI:
            interface = OpenAIModel(config)
        elif config.type == ModelType.VLLM:
            interface = VLLMModel(config)
        elif config.type == ModelType.LLAMA_CPP:
            interface = LlamaCppModel(config)
        elif config.type == ModelType.MOCK:
            interface = MockModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")
        
        return cls(interface)

    @classmethod
    def from_name(cls, name: str) -> "Model":
        """Create model from name (defaults to local OpenAI compatible model at localhost:1234)."""
        config = ModelConfig.from_name(name)
        return cls.from_config(config)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """Create model from dictionary."""
        config = ModelConfig.from_dict(data)
        return cls.from_config(config)

    def generate(self, prompt: str, global_sample_params: Optional[Dict[str, Any]] = None, **kwargs) -> SimpleInferenceResult:
        """Generate response for a prompt."""
        return self.interface.generate(prompt, global_sample_params, **kwargs)
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        return self.interface.is_available()
    
    def startup(self) -> None:
        """Start the model."""
        self.interface.startup()
    
    def shutdown(self) -> None:
        """Shutdown the model."""
        self.interface.shutdown()
    
    @property
    def name(self) -> str:
        """Get model name."""
        return self.config.name
    
    def __enter__(self):
        """Context manager entry."""
        self.startup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def create_model(model_input: Union[str, ModelConfig, Model, Dict[str, Any]]) -> Model:
    """
    Create a Model instance from various input types.
    
    Args:
        model_input: Can be:
            - str: Model name (defaults to local VLLM at localhost:1234)
            - ModelConfig: Model configuration object
            - Model: Returns as-is
            - Dict: Model configuration dictionary
    
    Returns:
        Model instance
    """
    if isinstance(model_input, Model):
        return model_input
    elif isinstance(model_input, str):
        return Model.from_name(model_input)
    elif isinstance(model_input, ModelConfig):
        return Model.from_config(model_input)
    elif isinstance(model_input, dict):
        return Model.from_dict(model_input)
    else:
        raise ValueError(f"Unsupported model input type: {type(model_input)}")


def create_models(models_input: List[Union[str, ModelConfig, Model, Dict[str, Any]]]) -> List[Model]:
    """
    Create Model instances from list of various input types.
    
    Args:
        models_input: List of model specifications
    
    Returns:
        List of Model instances
    """
    return [create_model(model_input) for model_input in models_input]
