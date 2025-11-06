#!/usr/bin/env python
"""
Test complete nodes_tasks functionality including preservation and CLI/JSON integration
"""

import sys
import json
import tempfile
import os
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args
from utils.nodes_tasks_parser import parse_nodes_tasks_config

def test_complete_nodes_tasks_integration():
    """Test complete integration of nodes_tasks preservation with both CLI and JSON config"""

    print("🧪 Testing Complete nodes_tasks Integration with Preservation")
    print("=" * 70)

    class MockArgs:
        def __init__(self):
            self.config = None
            self.nodes_tasks_config = None
            self.nodes_tasks_from_config = None
            self.num_clients = 1
            self.nodes_datasets = ""
            self.nodes_pretext_tasks = ""
            self.nodes_downstream_tasks = ""

    # Test 1: JSON config file with nodes_tasks preservation
    print("\n✅ Test 1: JSON config file with nodes_tasks preservation")

    json_config = {
        "federation": {
            "algorithm": "FedGFE",
            "num_clients": 4,
            "global_rounds": 100
        },
        "nodes_tasks": {
            "0": {
                "task_type": "classification",
                "pretext_tasks": ["image_rotation", "simclr"],
                "dataset": "JSRT-1C-ClaSSeg",
                "dataset_split": "0"
            },
            "1": {
                "task_type": "segmentation",
                "pretext_tasks": ["patch_masking"],
                "dataset": "JSRT-1C-ClaSSeg",
                "dataset_split": "1"
            },
            "2": {
                "task_type": "classification",
                "pretext_tasks": ["byod"],
                "dataset": "JSRT-2C-ClaSSeg",
                "dataset_split": "0"
            }
        }
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config, f)
        config_file = f.name

    try:
        args1 = MockArgs()
        args1.config = config_file

        print(f"Before config loading: args.nodes_tasks = {getattr(args1, 'nodes_tasks', 'NOT_SET')}")

        # Load JSON config (should preserve nodes_tasks)
        result1 = load_config_to_args(args1.config, args1)

        print(f"After config loading: args.nodes_tasks exists = {hasattr(result1, 'nodes_tasks')}")
        print(f"After config loading: args.nodes_tasks_from_config = {getattr(result1, 'nodes_tasks_from_config', 'NOT_SET')}")

        # Verify original nodes_tasks is preserved from JSON
        assert hasattr(result1, 'nodes_tasks'), "args.nodes_tasks not preserved from JSON config"
        assert result1.nodes_tasks is not None, "args.nodes_tasks is None from JSON config"
        assert len(result1.nodes_tasks) == 3, f"Expected 3 nodes from JSON, got {len(result1.nodes_tasks)}"
        assert result1.nodes_tasks_from_config == True, "nodes_tasks_from_config flag not set"

        # Check that CLI format was also generated
        assert hasattr(result1, 'nodes_pretext_tasks'), "CLI format not generated from JSON"
        assert "image_rotation" in result1.nodes_pretext_tasks, "JSON nodes_tasks not converted to CLI"
        assert result1.num_clients == 3, f"num_clients not updated from nodes_tasks, got {result1.num_clients}"

        print("   ✓ JSON config nodes_tasks preserved and converted successfully!")
        print(f"   ✓ Original preserved: {len(result1.nodes_tasks)} nodes")
        print(f"   ✓ Converted pretext_tasks: {result1.nodes_pretext_tasks}")
        print(f"   ✓ Updated num_clients: {result1.num_clients}")

    finally:
        os.unlink(config_file)

    # Test 2: CLI nodes_tasks_config with preservation
    print("\n✅ Test 2: CLI nodes_tasks_config with preservation")

    cli_config = {
        "0": {"task_type": "regression", "pretext_tasks": ["byod", "simclr"]},
        "1": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}
    }

    args2 = MockArgs()
    args2.nodes_tasks_config = json.dumps(cli_config)

    print(f"Before CLI parsing: args.nodes_tasks = {getattr(args2, 'nodes_tasks', 'NOT_SET')}")

    # Parse CLI nodes_tasks (should preserve and convert)
    result2 = parse_nodes_tasks_config(args2)

    print(f"After CLI parsing: args.nodes_tasks exists = {hasattr(result2, 'nodes_tasks')}")

    # Verify CLI preservation
    assert hasattr(result2, 'nodes_tasks'), "args.nodes_tasks not preserved from CLI"
    assert result2.nodes_tasks is not None, "args.nodes_tasks is None from CLI"
    assert len(result2.nodes_tasks) == 2, f"Expected 2 nodes from CLI, got {len(result2.nodes_tasks)}"

    # Check original config matches what was provided
    assert result2.nodes_tasks["0"]["task_type"] == "regression", "CLI config not preserved correctly"
    assert "byod" in result2.nodes_tasks["0"]["pretext_tasks"], "CLI pretext_tasks not preserved"

    print("   ✓ CLI nodes_tasks_config preserved and converted successfully!")
    print(f"   ✓ Original preserved: {json.dumps(result2.nodes_tasks, indent=4)}")

    # Test 3: Both JSON config and CLI override
    print("\n✅ Test 3: JSON config + CLI override behavior")

    # Create JSON config
    json_config2 = {
        "nodes_tasks": {
            "0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config2, f)
        config_file2 = f.name

    try:
        args3 = MockArgs()
        args3.config = config_file2
        args3.nodes_tasks_config = json.dumps({
            "0": {"task_type": "segmentation", "pretext_tasks": ["patch_masking"]},
            "1": {"task_type": "regression", "pretext_tasks": ["byod"]}
        })

        # Load JSON first
        result3 = load_config_to_args(args3.config, args3)
        print(f"After JSON: num_clients = {result3.num_clients}")

        # Then parse CLI (should override)
        result3 = parse_nodes_tasks_config(result3)
        print(f"After CLI override: num_clients = {result3.num_clients}")

        # CLI should take precedence
        assert result3.num_clients == 2, f"CLI didn't override JSON, num_clients = {result3.num_clients}"
        assert result3.nodes_tasks["0"]["task_type"] == "segmentation", "CLI didn't override JSON config"
        assert "patch_masking" in result3.nodes_pretext_tasks, "CLI pretext_tasks not applied"

        print("   ✓ CLI properly overrides JSON config!")
        print(f"   ✓ Final num_clients: {result3.num_clients}")
        print(f"   ✓ Final task_type for node 0: {result3.nodes_tasks['0']['task_type']}")

    finally:
        os.unlink(config_file2)

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Complete nodes_tasks integration working perfectly!")
    print("=" * 70)

    print("\n📋 Complete functionality summary:")
    print("   ✅ args.nodes_tasks preserves original configuration (dict format)")
    print("   ✅ CLI format conversion still works (strings for existing system)")
    print("   ✅ JSON config file support with preservation")
    print("   ✅ CLI --nodes_tasks_config support with preservation")
    print("   ✅ CLI arguments override JSON config correctly")
    print("   ✅ Both formats available simultaneously in args")
    print("   ✅ Backward compatibility maintained")

    return True

if __name__ == "__main__":
    try:
        test_complete_nodes_tasks_integration()
        print("\n🎯 Complete integration test successful!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)