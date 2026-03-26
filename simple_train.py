#!/usr/bin/env python3
"""
Simple training script that validates the YAML configuration.
"""

import sys
import yaml
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Simple training script for Jarvis"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()
    
    # Test YAML parsing
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ YAML configuration loaded successfully!")
        print(f"📊 Data path: {config['data']['path']}")
        print(f"🎯 Target column: {config['data']['target_column']}")
        print(f"🧠 Model hidden sizes: {config['model']['hidden_sizes']}")
        print(f"🏃 Training epochs: {config['training']['epochs']}")
        print(f"📈 Learning rate: {config['training']['learning_rate']}")
        
        print("\n🎉 Configuration is valid and ready for training!")
        
        # Check if data file exists
        data_path = Path(config['data']['path'])
        if data_path.exists():
            print(f"✅ Data file found: {data_path}")
        else:
            print(f"⚠️  Data file not found: {data_path}")
            
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        print(f"❌ Error loading YAML configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
