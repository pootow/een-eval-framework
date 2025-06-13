"""
Module loading utilities for the evaluation framework.

This module provides centralized utilities for loading functions, classes,
and other objects from external Python files and modules.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Callable, Any, Dict, Optional, Union


def load_function_from_file(file_path: str, function_name: str) -> Callable:
    """
    Load a function from a Python file.
    
    Args:
        file_path: Path to the Python file
        function_name: Name of the function to load
        
    Returns:
        The loaded function
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If the module cannot be loaded
        AttributeError: If the function doesn't exist in the module
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create a unique module name based on the file path to avoid conflicts
    module_name = f"custom_module_{path.stem}_{hash(file_path) & 0x7fffffff}"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {file_path}: {e}")
    
    if not hasattr(module, function_name):
        available_functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}. "
            f"Available functions: {available_functions}"
        )
    
    function = getattr(module, function_name)
    if not callable(function):
        raise AttributeError(f"'{function_name}' in {file_path} is not callable")
    
    return function


def load_function_from_module(module_name: str, function_name: str) -> Callable:
    """
    Load a function from an imported module.
    
    Args:
        module_name: Name of the module to import
        function_name: Name of the function to load
        
    Returns:
        The loaded function
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    
    if not hasattr(module, function_name):
        available_functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
        raise AttributeError(
            f"Function '{function_name}' not found in module '{module_name}'. "
            f"Available functions: {available_functions}"
        )
    
    function = getattr(module, function_name)
    if not callable(function):
        raise AttributeError(f"'{function_name}' in module '{module_name}' is not callable")
    
    return function


def load_object_from_file(file_path: str, object_name: str) -> Any:
    """
    Load any object (function, class, variable) from a Python file.
    
    Args:
        file_path: Path to the Python file
        object_name: Name of the object to load
        
    Returns:
        The loaded object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If the module cannot be loaded
        AttributeError: If the object doesn't exist in the module
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Create a unique module name based on the file path to avoid conflicts
    module_name = f"custom_module_{path.stem}_{hash(file_path) & 0x7fffffff}"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {file_path}: {e}")
    
    if not hasattr(module, object_name):
        available_objects = [name for name in dir(module) if not name.startswith('_')]
        raise AttributeError(
            f"Object '{object_name}' not found in {file_path}. "
            f"Available objects: {available_objects}"
        )
    
    return getattr(module, object_name)


def load_object_from_module(module_name: str, object_name: str) -> Any:
    """
    Load any object (function, class, variable) from an imported module.
    
    Args:
        module_name: Name of the module to import
        object_name: Name of the object to load
        
    Returns:
        The loaded object
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the object doesn't exist in the module
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    
    if not hasattr(module, object_name):
        available_objects = [name for name in dir(module) if not name.startswith('_')]
        raise AttributeError(
            f"Object '{object_name}' not found in module '{module_name}'. "
            f"Available objects: {available_objects}"
        )
    
    return getattr(module, object_name)


class CustomLoader:
    """
    A more advanced loader that can handle both file and module loading
    with configuration-based loading patterns.
    """
    
    @staticmethod
    def load_function(
        path: Optional[str] = None,
        module: Optional[str] = None,
        function_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Load a function from either a file or module based on configuration.
        
        Args:
            path: Path to Python file (for file-based loading)
            module: Module name (for module-based loading)
            function_name: Name of the function to load
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            The loaded function
            
        Raises:
            ValueError: If neither path nor module is provided
            Other exceptions from the underlying load functions
        """
        if not function_name:
            raise ValueError("function_name is required")
        
        if path:
            return load_function_from_file(path, function_name)
        elif module:
            return load_function_from_module(module, function_name)
        else:
            raise ValueError("Either 'path' or 'module' must be provided")
    
    @staticmethod
    def load_object(
        path: Optional[str] = None,
        module: Optional[str] = None,
        object_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Load any object from either a file or module based on configuration.
        
        Args:
            path: Path to Python file (for file-based loading)
            module: Module name (for module-based loading)
            object_name: Name of the object to load
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            The loaded object
            
        Raises:
            ValueError: If neither path nor module is provided
            Other exceptions from the underlying load functions
        """
        if not object_name:
            raise ValueError("object_name is required")
        
        if path:
            return load_object_from_file(path, object_name)
        elif module:
            return load_object_from_module(module, object_name)
        else:
            raise ValueError("Either 'path' or 'module' must be provided")
