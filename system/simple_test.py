#!/usr/bin/env python
"""
Simplified test to verify JSON loading works for basic parameters.
"""

import json
import sys
sys.path.append('/home/lpala/fedgfe/system')

def test_simple_json_loading():
    """Test basic JSON loading."""

    # Test loading the JSON file
    try:
        with open('configs/example_jsrt_8c.json', 'r') as f:
            config = json.load(f)

        print("✅ JSON file loaded successfully!")
        print(f"Algorithm: {config.get('federation', {}).get('algorithm', 'Not found')}")
        print(f"Num Clients: {config.get('federation', {}).get('num_clients', 'Not found')}")
        print(f"Global Rounds: {config.get('federation', {}).get('global_rounds', 'Not found')}")
        print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 'Not found')}")
        print(f"Dataset: {config.get('dataset', {}).get('name', 'Not found')}")

        return True

    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False

if __name__ == "__main__":
    test_simple_json_loading()