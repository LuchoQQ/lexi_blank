"""
Module for loading configuration from YAML files.
"""
import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if not config:
            config = {}
            
        # Ensure the configuration has the required sections
        for section in ['weaviate', 'neo4j', 'bm25', 'retrieval']:
            if section not in config:
                config[section] = {}
                
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the YAML configuration file
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        raise Exception(f"Error saving configuration: {str(e)}")
