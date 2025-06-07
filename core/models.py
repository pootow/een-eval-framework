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
import requests
import time


class ModelType(Enum):
    """Supported model types."""
    OPENAI = "openai"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    MOCK = "mock"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    docker_image: Optional[str] = "vllm/vllm-openai:latest"
    docker_params: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = -1
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    timeout: int = 300
    max_retries: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        model_type = ModelType(data.get("type", "openai"))
        return cls(
            name=data["name"],
            type=model_type,
            endpoint=data.get("endpoint"),
            api_key=data.get("api_key"),
            model_path=data.get("model_path"),
            docker_image=data.get("docker_image", "vllm/vllm-openai:latest"),
            docker_params=data.get("docker_params", {}),
            max_tokens=data.get("max_tokens", -1),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k", -1),
            timeout=data.get("timeout", 300),
            max_retries=data.get("max_retries", 3),
            extra_params=data.get("extra_params", {})
        )


@dataclass
class InferenceResult:
    """Result from model inference."""
    response: str
    prompt: str
    inference_time: float
    tokens_per_second: float
    model_name: str
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelInterface(ABC):
    """Abstract interface for model inference."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> InferenceResult:
        """Generate response for a prompt."""
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
    
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        # Merge config and override parameters
        params = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens) if self.config.max_tokens > 0 else None,
            **self.config.extra_params,
            **kwargs
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**params)
            end_time = time.time()
            
            inference_time = end_time - start_time
            content = response.choices[0].message.content
            
            # Calculate tokens per second if usage info is available
            total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
            tokens_per_second = total_tokens / inference_time if total_tokens else 0
            
            return InferenceResult(
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
            raise RuntimeError(f"OpenAI API call failed: {e}")
    
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
        
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate response using VLLM API."""
        if not self.client:
            raise RuntimeError("VLLM model not started. Call startup() first.")
        
        start_time = time.time()
        
        # Merge config and override parameters
        params = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens) if self.config.max_tokens > 0 else None,
            **self.config.extra_params,
            **kwargs
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = self.client.chat.completions.create(**params)
            end_time = time.time()
            
            inference_time = end_time - start_time
            content = response.choices[0].message.content
            
            # Calculate tokens per second if usage info is available
            total_tokens = getattr(response.usage, 'total_tokens', None) if response.usage else None
            tokens_per_second = total_tokens / inference_time if total_tokens else 0
            
            return InferenceResult(
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
            raise RuntimeError(f"VLLM API call failed: {e}")
    
    def is_available(self) -> bool:
        """Check if VLLM server is available."""
        if not self.config.endpoint:
            return False
        
        try:
            response = requests.get(f"{self.config.endpoint}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def startup(self) -> None:
        """Start VLLM server in Docker container."""
        from ..utils.docker import DockerManager
        
        if self.is_available():
            self.client = openai.OpenAI(base_url=self.config.endpoint)
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
        
        docker_params = {**default_params, **self.config.docker_params}

        # Validate required parameters
        if not self.config.model_path:
            raise ValueError("model_path is required for VLLM models")
        if not self.config.docker_image:
            raise ValueError("docker_image is required for VLLM models")
        
        self.docker_container_id = docker_manager.start_vllm_container(
            model_path=self.config.model_path,
            model_name=self.config.name,
            image=self.config.docker_image,
            host_port=self._extract_port_from_endpoint(),
            **docker_params
        )
        
        # Wait for server to be ready
        self._wait_for_server()
        
        # Setup client
        self.client = openai.OpenAI(base_url=self.config.endpoint)
    
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


class MockModel(ModelInterface):
    """Mock model implementation for testing purposes."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.call_count = 0
        self.responses = [
            "This is a mock response for testing purposes.",
            "Another mock response to simulate model behavior.",
            "Mock models are useful for testing the evaluation framework.",
            "Testing with mock models ensures reproducible results.",
            "The framework supports various model types including mocks."
        ]
    
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate mock response."""
        start_time = time.time()
        
        # Cycle through predefined responses
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        # Simulate some processing time
        time.sleep(0.1)
        end_time = time.time()
        
        inference_time = end_time - start_time
        mock_tokens = len(response.split()) + len(prompt.split())
        tokens_per_second = mock_tokens / inference_time if inference_time > 0 else 0
        
        return InferenceResult(
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
        elif config.type == ModelType.MOCK:
            interface = MockModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")
        
        return cls(interface)
    
    @classmethod
    def from_name(cls, name: str) -> "Model":
        """Create model from name (assumes OpenAI)."""
        config = ModelConfig(name=name, type=ModelType.OPENAI)
        return cls.from_config(config)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """Create model from dictionary."""
        config = ModelConfig.from_dict(data)
        return cls.from_config(config)
    
    def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate response for a prompt."""
        return self.interface.generate(prompt, **kwargs)
    
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
