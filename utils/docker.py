"""
Docker utilities for the evaluation framework.

This module handles Docker container management for VLLM models.
"""

import subprocess
import time
import logging
from typing import Dict, Any, Optional, List
import json


class DockerManager:
    """Manages Docker containers for VLLM models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_containers: Dict[str, str] = {}  # container_id -> model_name
    
    def start_vllm_container(
        self,
        model_path: str,
        model_name: str,
        image: str = "vllm/vllm-openai:latest",
        host_port: int = 8000,
        container_port: int = 8000,
        gpu_memory_utilization: float = 0.3,
        swap_space: int = 1,
        max_model_len: int = 8000,
        enable_reasoning: bool = True,
        reasoning_parser: str = "deepseek_r1",
        **kwargs
    ) -> str:
        """
        Start VLLM container.
        
        Returns:
            Container ID
        """
        # Build Docker command
        cmd = [
            "docker", "run", "-d",
            "--runtime", "nvidia",
            "--gpus", "all",
            "--ipc=host",
            "-p", f"{host_port}:{container_port}",
            image,
            f"--gpu-memory-utilization", str(gpu_memory_utilization),
            f"--swap-space", str(swap_space),
            f"--model", model_path,
            f"--max-model-len", str(max_model_len)
        ]
        
        # Add optional parameters
        if enable_reasoning:
            cmd.extend(["--enable-reasoning"])
            if reasoning_parser:
                cmd.extend(["--reasoning-parser", reasoning_parser])
        
        # Add any additional parameters
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        try:
            self.logger.info(f"Starting VLLM container for model: {model_name}")
            self.logger.debug(f"Docker command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            self.active_containers[container_id] = model_name
            
            self.logger.info(f"Started container {container_id} for model {model_name}")
            return container_id
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to start VLLM container: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def stop_container(self, container_id: str) -> None:
        """Stop Docker container."""
        try:
            self.logger.info(f"Stopping container: {container_id}")
            
            subprocess.run(
                ["docker", "stop", container_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Remove from active containers
            if container_id in self.active_containers:
                model_name = self.active_containers.pop(container_id)
                self.logger.info(f"Stopped container for model: {model_name}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop container {container_id}: {e.stderr}")
    
    def remove_container(self, container_id: str) -> None:
        """Remove Docker container."""
        try:
            self.logger.info(f"Removing container: {container_id}")
            
            subprocess.run(
                ["docker", "rm", container_id],
                capture_output=True,
                text=True,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to remove container {container_id}: {e.stderr}")
    
    def get_container_status(self, container_id: str) -> Optional[Dict[str, Any]]:
        """Get container status."""
        try:
            result = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            inspect_data = json.loads(result.stdout)
            if inspect_data:
                container_info = inspect_data[0]
                return {
                    "id": container_info["Id"][:12],
                    "name": container_info["Name"],
                    "state": container_info["State"]["Status"],
                    "running": container_info["State"]["Running"],
                    "created": container_info["Created"],
                    "ports": container_info.get("NetworkSettings", {}).get("Ports", {})
                }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to get container status: {e}")
        
        return None
    
    def wait_for_container_ready(
        self, 
        container_id: str, 
        endpoint: str = "http://localhost:8000",
        max_wait: int = 300,
        check_interval: int = 5
    ) -> bool:
        """Wait for container to be ready."""
        import requests
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check if container is still running
                status = self.get_container_status(container_id)
                if not status or not status["running"]:
                    self.logger.error(f"Container {container_id} is not running")
                    return False
                
                # Check if service is responding
                response = requests.get(f"{endpoint}/v1/models", timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"Container {container_id} is ready")
                    return True
                    
            except requests.RequestException:
                pass  # Service not ready yet
            
            time.sleep(check_interval)
        
        self.logger.error(f"Container {container_id} failed to become ready within {max_wait} seconds")
        return False
    
    def cleanup_all_containers(self) -> None:
        """Clean up all active containers."""
        for container_id in list(self.active_containers.keys()):
            try:
                self.stop_container(container_id)
                self.remove_container(container_id)
            except Exception as e:
                self.logger.error(f"Failed to cleanup container {container_id}: {e}")
    
    def list_vllm_containers(self) -> List[Dict[str, Any]]:
        """List all VLLM containers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "ancestor=vllm/vllm-openai", "--format", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            return containers
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to list VLLM containers: {e}")
            return []
    
    def get_container_logs(self, container_id: str, tail: int = 100) -> str:
        """Get container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get container logs: {e.stderr}")
            return ""
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup containers."""
        self.cleanup_all_containers()
