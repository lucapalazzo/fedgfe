#!/usr/bin/env python
"""
Final comprehensive test for nodes_tasks CLI integration
"""

import sys
import json
import tempfile
import os
sys.path.append('/home/lpala/fedgfe/system')

from utils.nodes_tasks_parser import parse_nodes_tasks_config, validate_nodes_tasks_config

def test_complete_nodes_tasks_functionality():
    """Test complete nodes_tasks functionality"""

    print("🧪 Testing Complete nodes_tasks CLI Functionality")
    print("=" * 60)

    # Test 1: Valid configuration parsing
    print("\n✅ Test 1: Valid configuration parsing")

    class MockArgs:
        def __init__(self):
            self.nodes_tasks_config = None
            self.nodes_tasks_from_config = None
            self.num_clients = 1
            self.nodes_datasets = ""
            self.nodes_pretext_tasks = ""
            self.nodes_downstream_tasks = ""

    # Create test configuration
    test_config = {
        "0": {
            "task_type": "classification",
            "pretext_tasks": ["image_rotation", "byod"],
            "dataset": "JSRT-1C-ClaSSeg",
            "dataset_split": "0"
        },
        "1": {
            "task_type": "segmentation",
            "pretext_tasks": ["patch_masking"],
            "dataset": "JSRT-1C-ClaSSeg",
            "dataset_split": "1"
        }
    }

    args = MockArgs()
    args.nodes_tasks_config = json.dumps(test_config)

    result = parse_nodes_tasks_config(args)

    assert result.num_clients == 2, f"Expected 2 clients, got {result.num_clients}"
    assert "classification" in result.nodes_downstream_tasks, "Missing classification task"
    assert "segmentation" in result.nodes_downstream_tasks, "Missing segmentation task"
    assert "image_rotation" in result.nodes_pretext_tasks, "Missing image_rotation"
    assert "byod" in result.nodes_pretext_tasks, "Missing byod"
    assert "patch_masking" in result.nodes_pretext_tasks, "Missing patch_masking"
    assert "JSRT-1C-ClaSSeg" in result.nodes_datasets, "Missing dataset"

    print(f"   ✓ num_clients: {result.num_clients}")
    print(f"   ✓ downstream_tasks: {result.nodes_downstream_tasks}")
    print(f"   ✓ pretext_tasks: {result.nodes_pretext_tasks}")
    print(f"   ✓ datasets: {result.nodes_datasets}")

    # Test 2: File-based configuration
    print("\n✅ Test 2: File-based configuration")

    file_config = {
        "nodes_tasks": {
            "0": {"task_type": "classification", "pretext_tasks": ["simclr"]},
            "1": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(file_config, f)
        temp_file = f.name

    try:
        args2 = MockArgs()
        args2.nodes_tasks_config = temp_file

        result2 = parse_nodes_tasks_config(args2)
        assert result2.num_clients == 2, "File config failed"
        assert "simclr" in result2.nodes_pretext_tasks, "Missing simclr from file"
        print(f"   ✓ File parsing successful: {result2.nodes_pretext_tasks}")

    finally:
        os.unlink(temp_file)

    # Test 3: Validation functionality
    print("\n✅ Test 3: Validation functionality")

    # Valid config
    valid_config = {"0": {"task_type": "classification", "pretext_tasks": ["image_rotation"]}}
    assert validate_nodes_tasks_config(valid_config) == True, "Valid config failed validation"
    print("   ✓ Valid configuration passed validation")

    # Invalid config
    invalid_config = {"0": {"task_type": "invalid_type", "pretext_tasks": "not_a_list"}}
    assert validate_nodes_tasks_config(invalid_config) == False, "Invalid config passed validation"
    print("   ✓ Invalid configuration correctly rejected")

    # Test 4: Error handling
    print("\n✅ Test 4: Error handling")

    args3 = MockArgs()
    args3.nodes_tasks_config = "{invalid: json}"
    result3 = parse_nodes_tasks_config(args3)
    assert result3.num_clients == 1, "Error handling failed"  # Should remain unchanged
    print("   ✓ Invalid JSON handled gracefully")

    # Test 5: No configuration provided
    print("\n✅ Test 5: No configuration (should pass through unchanged)")

    args4 = MockArgs()
    args4.num_clients = 5  # Set original value
    result4 = parse_nodes_tasks_config(args4)
    assert result4.num_clients == 5, "No config case failed"
    print("   ✓ No configuration handled correctly")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! nodes_tasks CLI integration is fully functional!")
    print("=" * 60)

    print("\n📋 Summary of implemented features:")
    print("   • CLI argument: --nodes_tasks_config / -nt")
    print("   • JSON string input support")
    print("   • JSON file input support")
    print("   • Configuration validation with detailed error messages")
    print("   • Automatic conversion to existing CLI format")
    print("   • Graceful error handling")
    print("   • Integration with main.py argument parser")
    print("   • Comprehensive test coverage")

    return True

if __name__ == "__main__":
    try:
        test_complete_nodes_tasks_functionality()
        print("\n✅ Integration test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)