"""
Dataset handling for the evaluation framework.

This module provides dataset loading and processing capabilities
for various formats used in model evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
import json
import csv
import importlib.util
from pathlib import Path

from ..utils.module_loader import load_function_from_file, load_function_from_module


@dataclass
class DatasetItem:
    """Single item in a dataset."""
    id: str
    data: Dict[str, Any]
    
    def __getitem__(self, key: str) -> Any:
        return self.data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()


class Dataset:
    """Dataset container for evaluation."""
    
    def __init__(self, items: List[DatasetItem], metadata: Optional[Dict[str, Any]] = None):
        self.items = items
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[DatasetItem, List[DatasetItem]]:
        return self.items[index]
    
    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> "Dataset":
        """Load dataset from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if path.suffix.lower() == '.jsonl':
            return cls._load_jsonl(file_path, **kwargs)
        elif path.suffix.lower() == '.json':
            return cls._load_json(file_path, **kwargs)
        elif path.suffix.lower() in ['.csv', '.tsv']:
            return cls._load_csv(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "Dataset":
        """Create dataset from dictionary."""
        if "items" in data:
            items_data = data["items"]
            metadata = {k: v for k, v in data.items() if k != "items"}
        else:
            items_data = data
            metadata = {}
        
        items = []
        for i, item_data in enumerate(items_data):
            if isinstance(item_data, dict):
                item_id = item_data.get("id", str(i))
                items.append(DatasetItem(id=item_id, data=item_data))
            else:
                items.append(DatasetItem(id=str(i), data={"value": item_data}))
        
        return cls(items, metadata)
    
    @classmethod
    def from_list(cls, data: List[Any], **kwargs) -> "Dataset":
        """Create dataset from list."""
        items = []
        for i, item_data in enumerate(data):
            if isinstance(item_data, dict):
                item_id = item_data.get("id", str(i))
                items.append(DatasetItem(id=item_id, data=item_data))
            else:
                items.append(DatasetItem(id=str(i), data={"value": item_data}))
        
        return cls(items)
    
    @classmethod
    def from_function(
        cls,
        function: Callable,
        **params
    ) -> "Dataset":
        """Create dataset from custom function."""
        result = function(**params)
        
        if isinstance(result, cls):            return result
        elif isinstance(result, list):
            return cls.from_list(result)
        elif isinstance(result, dict):
            return cls.from_dict(result)
        else:
            raise ValueError(f"Custom dataset function must return Dataset, list, or dict, got {type(result)}")
    
    @classmethod
    def from_file_function(
        cls,
        file_path: str, 
        function_name: str,
        **params
    ) -> "Dataset":
        """Create dataset from function in external file."""
        if not function_name:
            raise ValueError("function_name is required for custom dataset loading")
        function = load_function_from_file(file_path, function_name)
        return cls.from_function(function, **params)
    
    @classmethod
    def from_config(cls, config) -> "Dataset":
        """Create dataset from DatasetConfig object."""
        # Import here to avoid circular imports
        from ..workflow.config import DatasetConfig
        
        if not isinstance(config, DatasetConfig):
            raise ValueError("Expected DatasetConfig object")
        
        if config.type == "built_in":
            # Built-in dataset loading (from file paths)
            if config.path:
                return cls.from_file(config.path)
            else:
                raise ValueError("Built-in dataset type requires a path")
        elif config.type == "custom":
            if config.path:
                # File-based custom dataset function
                if not config.function_name:
                    raise ValueError("function_name is required for custom dataset type")
                return cls.from_file_function(config.path, config.function_name, **config.params)
            elif config.module and config.function_name:
                # Module-based custom dataset function
                try:
                    function = load_function_from_module(config.module, config.function_name)
                    return cls.from_function(function, **config.params)
                except Exception as e:
                    raise ImportError(f"Could not load function {config.function_name} from module {config.module}: {e}")
            else:
                raise ValueError("Custom dataset type requires either path or module/function_name")
        elif config.type == "file":
            if config.path:
                return cls.from_file(config.path)
            else:
                raise ValueError("File dataset type requires a path")
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")
    
    @classmethod
    def _load_jsonl(cls, file_path: str, **kwargs) -> "Dataset":
        """Load JSONL dataset."""
        items = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    item_id = data.get("id", str(i))
                    items.append(DatasetItem(id=item_id, data=data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i+1}: {e}")
        
        metadata = {
            "file_path": file_path,
            "format": "jsonl",
            "total_items": len(items)
        }
        
        return cls(items, metadata)
    
    @classmethod
    def _load_json(cls, file_path: str, **kwargs) -> "Dataset":
        """Load JSON dataset."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            items = []
            for i, item_data in enumerate(data):
                if isinstance(item_data, dict):
                    item_id = item_data.get("id", str(i))
                    items.append(DatasetItem(id=item_id, data=item_data))
                else:
                    items.append(DatasetItem(id=str(i), data={"value": item_data}))
        elif isinstance(data, dict):
            if "items" in data:
                items_data = data["items"]
                metadata_extra = {k: v for k, v in data.items() if k != "items"}
            else:
                items_data = [data]
                metadata_extra = {}
            
            items = []
            for i, item_data in enumerate(items_data):
                if isinstance(item_data, dict):
                    item_id = item_data.get("id", str(i))
                    items.append(DatasetItem(id=item_id, data=item_data))
                else:
                    items.append(DatasetItem(id=str(i), data={"value": item_data}))
        else:
            raise ValueError("JSON data must be a list or dict")
        
        metadata = {
            "file_path": file_path,
            "format": "json",
            "total_items": len(items),
            **metadata_extra
        }
        
        return cls(items, metadata)
    
    @classmethod
    def _load_csv(cls, file_path: str, **kwargs) -> "Dataset":
        """Load CSV/TSV dataset."""
        delimiter = '\t' if file_path.endswith('.tsv') else ','
        delimiter = kwargs.get('delimiter', delimiter)
        
        items = []
        
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                item_id = row.get("id", str(i))
                items.append(DatasetItem(id=item_id, data=dict(row)))
        
        metadata = {
            "file_path": file_path,
            "format": "csv" if delimiter == ',' else "tsv",
            "total_items": len(items),
            "delimiter": delimiter
        }
        
        return cls(items, metadata)
    
    def save(self, file_path: str, format: Optional[str] = None) -> None:
        """Save dataset to file."""
        path = Path(file_path)
        
        if format is None:
            format = path.suffix.lower()[1:]  # Remove the dot
        
        if format == 'jsonl':
            self._save_jsonl(file_path)
        elif format == 'json':
            self._save_json(file_path)
        elif format in ['csv', 'tsv']:
            self._save_csv(file_path, format)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_jsonl(self, file_path: str) -> None:
        """Save as JSONL format."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in self.items:
                json.dump(item.data, f, ensure_ascii=False)
                f.write('\n')
    
    def _save_json(self, file_path: str) -> None:
        """Save as JSON format."""
        data = {
            "items": [item.data for item in self.items],
            **self.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, file_path: str, format: str) -> None:
        """Save as CSV/TSV format."""
        delimiter = '\t' if format == 'tsv' else ','
        
        if not self.items:
            return
        
        # Get all possible fieldnames
        fieldnames = set()
        for item in self.items:
            fieldnames.update(item.data.keys())
        
        fieldnames = sorted(fieldnames)
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            
            for item in self.items:
                writer.writerow(item.data)
    
    def filter(self, condition: callable) -> "Dataset":
        """Filter dataset items based on condition."""
        filtered_items = [item for item in self.items if condition(item)]
        return Dataset(filtered_items, self.metadata.copy())
    
    def map(self, transform: callable) -> "Dataset":
        """Transform dataset items."""
        transformed_items = []
        for item in self.items:
            new_data = transform(item.data)
            transformed_items.append(DatasetItem(id=item.id, data=new_data))
        
        return Dataset(transformed_items, self.metadata.copy())
    
    def sample(self, n: int, random_seed: Optional[int] = None) -> "Dataset":
        """Sample n items from dataset."""
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        if n >= len(self.items):
            return Dataset(self.items.copy(), self.metadata.copy())
        
        sampled_items = random.sample(self.items, n)
        metadata = self.metadata.copy()
        metadata["sampled"] = True
        metadata["sample_size"] = n
        metadata["original_size"] = len(self.items)
        
        return Dataset(sampled_items, metadata)
    
    def split(self, train_ratio: float = 0.8, random_seed: Optional[int] = None) -> tuple["Dataset", "Dataset"]:
        """Split dataset into train and test sets."""
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        items_copy = self.items.copy()
        random.shuffle(items_copy)
        
        split_index = int(len(items_copy) * train_ratio)
        
        train_items = items_copy[:split_index]
        test_items = items_copy[split_index:]
        
        train_metadata = self.metadata.copy()
        train_metadata["split"] = "train"
        train_metadata["split_ratio"] = train_ratio
        
        test_metadata = self.metadata.copy()
        test_metadata["split"] = "test"
        test_metadata["split_ratio"] = 1 - train_ratio
        
        return Dataset(train_items, train_metadata), Dataset(test_items, test_metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.items:
            return {"total_items": 0}
        
        # Basic stats
        stats = {
            "total_items": len(self.items),
            "metadata": self.metadata
        }
        
        # Field analysis
        field_counts = {}
        field_types = {}

        for item in self.items:
            for key, value in item.data.items():
                field_counts[key] = field_counts.get(key, 0) + 1
                
                value_type = type(value).__name__
                if key not in field_types:
                    field_types[key] = set()
                field_types[key].add(value_type)
        
        stats["fields"] = {
            "counts": field_counts,
            "types": {k: list(v) for k, v in field_types.items()}
        }
        
        return stats
