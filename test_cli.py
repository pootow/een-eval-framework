#!/usr/bin/env python3
"""
Test script to verify the een_eval CLI functionality.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return e


def test_cli_commands():
    """Test basic CLI commands."""
    print("Testing een_eval CLI commands...")
    print("=" * 50)
    
    # Test help
    print("\n1. Testing help command:")
    run_command([sys.executable, "-m", "een_eval.cli", "--help"], check=False)
    
    # Test list components
    print("\n2. Testing list-components:")
    run_command([sys.executable, "-m", "een_eval.cli", "list-components"], check=False)
    
    # Test create config
    print("\n3. Testing create-config:")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    run_command([
        sys.executable, "-m", "een_eval.cli", 
        "create-config", 
        "--output", str(output_dir / "test_config.yaml"),
        "--type", "basic"
    ], check=False)
    
    # Test validate config (if file was created)
    config_file = output_dir / "test_config.yaml"
    if config_file.exists():
        print("\n4. Testing validate:")
        run_command([
            sys.executable, "-m", "een_eval.cli",
            "validate",
            "--config", str(config_file)
        ], check=False)
    
    print("\n" + "=" * 50)
    print("CLI testing completed!")


def check_imports():
    """Check if all imports work correctly."""
    print("Testing imports...")
    print("=" * 30)
    
    try:
        import een_eval
        print("✅ een_eval package imported successfully")
        
        from een_eval import EvalWorkflow
        print("✅ EvalWorkflow imported successfully")
        
        from een_eval.cli import main
        print("✅ CLI main function imported successfully")
        
        from een_eval.core.models import Model, ModelConfig
        print("✅ Core models imported successfully")
        
        from een_eval.core.evaluation import EvaluationMethod
        print("✅ Evaluation methods imported successfully")
        
        from een_eval.core.metrics import Metric
        print("✅ Metrics imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("een_eval Framework Test Script")
    print("=" * 40)
    
    # Test imports first
    if not check_imports():
        print("❌ Import tests failed. Exiting.")
        return 1
    
    # Test CLI
    test_cli_commands()
    
    print("\n🎉 Testing completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
