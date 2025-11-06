#!/usr/bin/env python
"""
Test that nodes_tasks original configuration is preserved in args.nodes_tasks
"""

import sys
import json
sys.path.append('/home/lpala/fedgfe/system')

from utils.nodes_tasks_parser import parse_nodes_tasks_config

def test_nodes_tasks_preservation():
    """Test that original nodes_tasks config is preserved in args.nodes_tasks"""

    print("🧪 Testing nodes_tasks Configuration Preservation")
    print("=" * 60)

    class MockArgs:
        def __init__(self):
            self.nodes_tasks_config = None
            self.nodes_tasks_from_config = None
            self.num_clients = 1
            self.nodes_datasets = ""
            self.nodes_pretext_tasks = ""
            self.nodes_downstream_tasks = ""

    # Test 1: CLI JSON string configuration
    print("\n✅ Test 1: Original configuration preservation from CLI")

    original_config = {
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
    args.nodes_tasks_config = json.dumps(original_config)

    print(f"Before parsing: args.nodes_tasks = {getattr(args, 'nodes_tasks', 'NOT_SET')}")

    result = parse_nodes_tasks_config(args)

    print(f"After parsing: args.nodes_tasks exists = {hasattr(result, 'nodes_tasks')}")

    # Verify original config is preserved
    assert hasattr(result, 'nodes_tasks'), "args.nodes_tasks not created"
    assert result.nodes_tasks is not None, "args.nodes_tasks is None"
    assert isinstance(result.nodes_tasks, dict), "args.nodes_tasks is not a dict"

    # Verify content matches
    assert len(result.nodes_tasks) == 2, f"Expected 2 nodes, got {len(result.nodes_tasks)}"
    assert "0" in result.nodes_tasks, "Node 0 missing from preserved config"
    assert "1" in result.nodes_tasks, "Node 1 missing from preserved config"

    # Check specific node content
    node_0 = result.nodes_tasks["0"]
    assert node_0["task_type"] == "classification", "Node 0 task_type not preserved"
    assert "image_rotation" in node_0["pretext_tasks"], "Node 0 pretext_tasks not preserved"
    assert node_0["dataset"] == "JSRT-1C-ClaSSeg", "Node 0 dataset not preserved"

    node_1 = result.nodes_tasks["1"]
    assert node_1["task_type"] == "segmentation", "Node 1 task_type not preserved"
    assert "patch_masking" in node_1["pretext_tasks"], "Node 1 pretext_tasks not preserved"

    print("   ✓ Original configuration perfectly preserved!")
    print(f"   ✓ Preserved nodes_tasks: {json.dumps(result.nodes_tasks, indent=2)}")

    # Test 2: No configuration case
    print("\n✅ Test 2: No configuration provided")

    args2 = MockArgs()
    result2 = parse_nodes_tasks_config(args2)

    assert hasattr(result2, 'nodes_tasks'), "args.nodes_tasks not initialized"
    assert result2.nodes_tasks is None, "args.nodes_tasks should be None when no config"

    print("   ✓ args.nodes_tasks correctly initialized to None")

    # Test 3: Verify both original and converted formats coexist
    print("\n✅ Test 3: Both original and converted formats available")

    args3 = MockArgs()
    args3.nodes_tasks_config = json.dumps({
        "0": {"task_type": "classification", "pretext_tasks": ["simclr"]},
        "1": {"task_type": "regression", "pretext_tasks": ["byod"]}
    })

    result3 = parse_nodes_tasks_config(args3)

    # Check original format is preserved
    assert result3.nodes_tasks is not None, "Original config not preserved"
    assert len(result3.nodes_tasks) == 2, "Original config incomplete"

    # Check converted format is available
    assert hasattr(result3, 'nodes_pretext_tasks'), "Converted pretext_tasks missing"
    assert hasattr(result3, 'nodes_downstream_tasks'), "Converted downstream_tasks missing"
    assert result3.num_clients == 2, "num_clients not updated"

    print("   ✓ Original format preserved in args.nodes_tasks")
    print(f"   ✓ Converted pretext_tasks: {result3.nodes_pretext_tasks}")
    print(f"   ✓ Converted downstream_tasks: {result3.nodes_downstream_tasks}")
    print(f"   ✓ Updated num_clients: {result3.num_clients}")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! nodes_tasks preservation is working correctly!")
    print("=" * 60)

    print("\n📋 Now args contains BOTH formats:")
    print("   • args.nodes_tasks: Original per-node configuration dict")
    print("   • args.nodes_datasets: Converted CLI format string")
    print("   • args.nodes_pretext_tasks: Converted CLI format string")
    print("   • args.nodes_downstream_tasks: Converted CLI format string")
    print("   • args.num_clients: Updated count from nodes_tasks")

    return True

if __name__ == "__main__":
    try:
        test_nodes_tasks_preservation()
        print("\n✅ Preservation test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)