#!/usr/bin/env python
"""
Test patch_count calculation from JSON model configuration
"""

import sys
import json
import tempfile
import os
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_patch_count_calculation():
    """Test automatic patch_count calculation from patch_size and dataset_image_size."""

    print("🧪 Testing patch_count Calculation from JSON Model Configuration")
    print("=" * 70)

    class MockArgs:
        def __init__(self):
            # Default args (use defaults that won't interfere with JSON mapping)
            self.config = None
            self.dataset_image_size = -1  # Use default value that config_loader uses
            self.patch_size = 16  # Use actual default value from main.py
            self.patch_count = None
            self.model = "cnn"  # Default that will be overridden
            self.nodes_backbone_model = "hf_vit"
            self.num_classes = 10
            self.embedding_size = 768

    # Test 1: Automatic calculation from JSON
    print("\n✅ Test 1: Automatic patch_count calculation")

    json_config = {
        "model": {
            "backbone": "vit",
            "backbone_model": "hf_vit",
            "patch_size": 16,
            "num_classes": 2,
            "embedding_size": 768
        },
        "dataset": {
            "image_size": 224
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config, f)
        config_file = f.name

    try:
        args = MockArgs()
        args.config = config_file

        print(f"Before loading: patch_count = {args.patch_count}")
        print(f"Before loading: dataset_image_size = {args.dataset_image_size}")
        print(f"Before loading: patch_size = {args.patch_size}")

        # Load config
        result = load_config_to_args(args.config, args)

        print(f"After loading: patch_count = {getattr(result, 'patch_count', 'NOT_SET')}")
        print(f"After loading: dataset_image_size = {result.dataset_image_size}")
        print(f"After loading: patch_size = {result.patch_size}")

        # Verify calculation: (224 / 16)² = 14² = 196
        expected_patch_count = (224 // 16) ** 2
        assert hasattr(result, 'patch_count'), "patch_count not calculated"
        assert result.patch_count == expected_patch_count, f"Expected {expected_patch_count}, got {result.patch_count}"

        print(f"   ✓ Automatic calculation successful: {result.patch_count} patches")
        print(f"   ✓ Formula: ({result.dataset_image_size} / {result.patch_size})² = {result.patch_count}")

    finally:
        os.unlink(config_file)

    # Test 2: Explicit patch_count in JSON
    print("\n✅ Test 2: Explicit patch_count specification")

    json_config2 = {
        "model": {
            "backbone": "vit",
            "patch_size": 8,
            "patch_count": 500  # Explicit value
        },
        "dataset": {
            "image_size": 224
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_config2, f)
        config_file2 = f.name

    try:
        args2 = MockArgs()
        args2.config = config_file2

        result2 = load_config_to_args(args2.config, args2)

        # Explicit value should take precedence
        assert result2.patch_count == 500, f"Expected explicit value 500, got {result2.patch_count}"
        print(f"   ✓ Explicit patch_count preserved: {result2.patch_count}")

    finally:
        os.unlink(config_file2)

    # Test 3: Different image sizes and patch sizes
    print("\n✅ Test 3: Different combinations")

    test_cases = [
        {"img_size": 224, "patch_size": 16, "expected": 196},  # (224/16)² = 14² = 196
        {"img_size": 224, "patch_size": 8, "expected": 784},   # (224/8)² = 28² = 784
        {"img_size": 256, "patch_size": 16, "expected": 256},  # (256/16)² = 16² = 256
        {"img_size": 384, "patch_size": 16, "expected": 576},  # (384/16)² = 24² = 576
    ]

    for i, test_case in enumerate(test_cases):
        json_config3 = {
            "model": {"patch_size": test_case["patch_size"]},
            "dataset": {"image_size": test_case["img_size"]}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_config3, f)
            config_file3 = f.name

        try:
            args3 = MockArgs()
            args3.config = config_file3

            result3 = load_config_to_args(args3.config, args3)

            expected = test_case["expected"]
            actual = getattr(result3, 'patch_count', None)
            actual_img_size = getattr(result3, 'dataset_image_size', None)
            actual_patch_size = getattr(result3, 'patch_size', None)

            print(f"   Debug Case {i+1}: img_size={actual_img_size}, patch_size={actual_patch_size}, patch_count={actual}")

            assert actual == expected, f"Case {i+1}: Expected {expected}, got {actual} (img_size={actual_img_size}, patch_size={actual_patch_size})"
            print(f"   ✓ Case {i+1}: {test_case['img_size']}x{test_case['img_size']} ÷ {test_case['patch_size']}² = {actual} patches")

        finally:
            os.unlink(config_file3)

    print("\n" + "=" * 70)
    print("🎉 All patch_count calculation tests passed!")
    print("=" * 70)

    print("\n📋 Summary of features:")
    print("   ✅ Automatic patch_count calculation from JSON model.patch_size")
    print("   ✅ Uses dataset.image_size for calculation")
    print("   ✅ Formula: (image_size / patch_size)²")
    print("   ✅ Explicit patch_count specification supported")
    print("   ✅ Explicit values take precedence over calculation")
    print("   ✅ Multiple image/patch size combinations supported")

    return True

if __name__ == "__main__":
    try:
        test_patch_count_calculation()
        print("\n🎯 patch_count calculation test successful!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)