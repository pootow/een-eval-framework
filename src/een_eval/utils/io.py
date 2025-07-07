"""
I/O utilities for the evaluation framework.

This module handles file operations, output management, and result persistence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import logging

from ..core.models import InferenceResult


class OutputManager:
    """Manages output files and directories for evaluation workflow."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize output files
        self.inference_file = self.output_dir / "inference.json"
        self.responses_file = self.output_dir / "responses.jsonl"
        self.evaluation_file = self.output_dir / "evaluation_results.jsonl"
        self.status_file = self.output_dir / "status.json"
        self.metrics_file = self.output_dir / "metrics.json"
        self.config_file = self.output_dir / "config.json"
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def save_inference_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save inference metadata."""
        try:
            # Overwrite inference file
            with open(self.inference_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                f.write('\n')
            self.logger.debug("Saved inference metadata")
        except Exception as e:
            self.logger.error(f"Failed to save inference metadata: {e}")
    
    def save_response(self, response_data: Dict[str, Any]) -> None:
        """Save single response to responses file."""
        try:
            # Append to responses file
            with open(self.responses_file, 'a', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, default=str)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save response: {e}")

    def save_responses_batch(self, responses: List[InferenceResult]) -> None:
        """Save batch of responses."""
        try:
            with open(self.responses_file, 'a', encoding='utf-8') as f:
                for response in responses:
                    # Convert dataclass to dict if needed
                    response_dict = {}
                    if hasattr(response, '__dataclass_fields__'):
                        # It's a dataclass, convert to dict
                        try:
                            from dataclasses import asdict
                            response_dict = asdict(response)
                        except Exception:
                            # Fallback to manual conversion
                            response_dict = {field: getattr(response, field, None) 
                                           for field in response.__dataclass_fields__}
                    elif isinstance(response, dict):
                        response_dict = response
                    else:
                        # Fallback: try to convert to dict
                        response_dict = response.__dict__ if hasattr(response, '__dict__') else {"value": str(response)}
                    json.dump(response_dict, f, ensure_ascii=False, default=str)
                    f.write('\n')
            self.logger.debug(f"Saved {len(responses)} responses")
        except Exception as e:
            self.logger.error(f"Failed to save responses batch: {e}")
    
    def load_responses(self) -> List[Dict[str, Any]]:
        """Load all responses from file."""
        responses = []
        
        if not self.responses_file.exists():
            return responses
        
        try:
            with open(self.responses_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        responses.append(json.loads(line))
            
            self.logger.debug(f"Loaded {len(responses)} responses")
            return responses
            
        except Exception as e:
            self.logger.error(f"Failed to load responses: {e}")
            return []
    
    def save_evaluation_results(self, results: list) -> None:
        """Save evaluation results."""
        try:
            with open(self.evaluation_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False, default=str)
                    f.write('\n')
            self.logger.debug(f"Saved evaluation results to {self.evaluation_file}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")
    
    def load_evaluation_results(self) -> Optional[Dict[str, Any]]:
        """Load evaluation results."""
        if not self.evaluation_file.exists():
            return None
        
        try:
            with open(self.evaluation_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            self.logger.debug("Loaded evaluation results")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load evaluation results: {e}")
            return None
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics."""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self) -> Optional[Dict[str, Any]]:
        """Load metrics."""
        if not self.metrics_file.exists():
            return None
        
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            self.logger.debug("Loaded metrics")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            return None
    
    def save_status(self, status: Any) -> None:
        """Save workflow status."""
        try:
            if hasattr(status, 'to_dict'):
                status_data = status.to_dict()
            else:
                status_data = status
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug("Saved workflow status")
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")
    
    def load_status(self) -> Optional[Dict[str, Any]]:
        """Load workflow status."""
        if not self.status_file.exists():
            return None
        
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            self.logger.debug("Loaded workflow status")
            return status
        except Exception as e:
            self.logger.error(f"Failed to load status: {e}")
            return None
    
    def save_intermediate_result(self, name: str, data: Any) -> None:
        """Save intermediate result with custom name."""
        try:
            file_path = self.output_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug(f"Saved intermediate result: {name}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate result {name}: {e}")
    
    def load_intermediate_result(self, name: str) -> Optional[Any]:
        """Load intermediate result by name."""
        file_path = self.output_dir / f"{name}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.debug(f"Loaded intermediate result: {name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load intermediate result {name}: {e}")
            return None
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of output files."""
        summary = {
            "output_dir": str(self.output_dir),
            "files": {}
        }
        
        files_to_check = [
            ("config", self.config_file),
            ("inference", self.inference_file),
            ("responses", self.responses_file),
            ("evaluation", self.evaluation_file),
            ("metrics", self.metrics_file),
            ("status", self.status_file)
        ]
        
        for name, file_path in files_to_check:
            if file_path.exists():
                stat = file_path.stat()
                summary["files"][name] = {
                    "path": str(file_path),
                    "exists": True,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                summary["files"][name] = {
                    "path": str(file_path),
                    "exists": False
                }
        
        return summary
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        temp_patterns = ["*.tmp", "*.temp", ".lock"]
        
        for pattern in temp_patterns:
            for file_path in self.output_dir.glob(pattern):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def archive_results(self, archive_name: Optional[str] = None) -> str:
        """Archive all results into a compressed file."""
        import shutil
        
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"eval_results_{timestamp}"
        
        archive_path = self.output_dir.parent / f"{archive_name}.tar.gz"
        
        try:
            shutil.make_archive(
                str(archive_path.with_suffix('')),
                'gztar',
                self.output_dir.parent,
                self.output_dir.name
            )
            self.logger.info(f"Archived results to: {archive_path}")
            return str(archive_path)
        except Exception as e:
            self.logger.error(f"Failed to archive results: {e}")
            raise
    
    def load_results(self) -> Optional[Dict[str, Any]]:
        """Load all available results."""
        results = {}
        
        # Load evaluation results
        eval_results = self.load_evaluation_results()
        if eval_results:
            results["evaluation"] = eval_results
        
        # Load metrics
        metrics = self.load_metrics()
        if metrics:
            results["metrics"] = metrics
        
        # Load status
        status = self.load_status()
        if status:
            results["status"] = status
        
        # Load summary
        results["summary"] = self.get_output_summary()
        
        return results if results else None
    
    def save_evaluation_summary(self, summary: Dict[str, Any]) -> None:
        """Save evaluation summary."""
        try:
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            self.logger.debug(f"Saved evaluation summary to {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation summary: {e}")
    
    def clean_output_files(self) -> None:
        """Clean output files to start fresh (used when resume=False)."""
        files_to_clean = [
            self.inference_file,
            self.responses_file,
            self.evaluation_file,
            self.metrics_file,
            self.status_file
        ]
        
        for file_path in files_to_clean:
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned output file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean output file {file_path}: {e}")
        
        # Also clean up any intermediate result files
        for file_path in self.output_dir.glob("*.json"):
            if file_path.name not in ["config.json"]:  # Keep config file
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned intermediate file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean intermediate file {file_path}: {e}")
        
        self.logger.info("Output files cleaned for fresh start")
    
    def save_failed_responses(self, failed_responses: List[Dict[str, Any]]) -> None:
        """Save failed responses to a separate file."""
        try:
            failed_file = self.output_dir / "failed_responses.jsonl"
            with open(failed_file, 'a', encoding='utf-8') as f:
                for response in failed_responses:
                    json.dump(response, f, ensure_ascii=False, default=str)
                    f.write('\n')
            self.logger.debug(f"Appended {len(failed_responses)} failed responses to {failed_file}")
        except Exception as e:
            self.logger.error(f"Failed to save failed responses: {e}")
    
    def rewrite_responses_file(self, responses: List[Dict[str, Any]]) -> None:
        """Rewrite responses.jsonl with provided responses."""
        try:
            with open(self.responses_file, 'w', encoding='utf-8') as f:
                for response in responses:
                    json.dump(response, f, ensure_ascii=False, default=str)
                    f.write('\n')
            self.logger.debug(f"Rewrote responses.jsonl with {len(responses)} responses")
        except Exception as e:
            self.logger.error(f"Failed to rewrite responses file: {e}")
    
    def clear_responses_file(self) -> None:
        """Clear the responses.jsonl file, backing up the old file first."""
        try:
            # Backup if file exists and is not empty
            if self.responses_file.exists() and self.responses_file.stat().st_size > 0:
                backup_idx = 1
                backup_file = self.output_dir / "responses.jsonl.bak"
                # Find next available backup filename
                while backup_file.exists():
                    backup_file = self.output_dir / f"responses.jsonl.bak{backup_idx}"
                    backup_idx += 1
                self.responses_file.rename(backup_file)
                self.logger.debug(f"Backed up responses.jsonl to {backup_file}")
            # Now clear (create empty file)
            with open(self.responses_file, 'w', encoding='utf-8') as f:
                pass  # Just create empty file
            self.logger.debug("Cleared responses.jsonl file")
        except Exception as e:
            self.logger.error(f"Failed to clear responses file: {e}")
