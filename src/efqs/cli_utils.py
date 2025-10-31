"""
Common utilities for CLI scripts: config loading, argument parsing helpers, etc.
"""
from __future__ import annotations
import json
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file.
    
    path: path to config file
    Returns: config dict
    """
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    
    # Try YAML
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install PyYAML")
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {path}: {e}")


def save_results_json(results: Any, path: str, indent: int = 2):
    """Save results to a JSON file.
    
    results: data structure to save (must be JSON-serializable)
    path: output file path
    indent: JSON indentation level
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=indent)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config.
    
    base: base configuration
    override: values to override
    Returns: merged config (base is not modified)
    """
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_configs(merged[key], val)
        else:
            merged[key] = val
    return merged
