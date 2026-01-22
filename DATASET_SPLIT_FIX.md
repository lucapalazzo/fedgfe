# Dataset Split Validation Fix

## Problem

The dataset splitting logic was failing with this error:

```
sklearn.utils._param_validation.InvalidParameterError:
The 'test_size' parameter of train_test_split must be a float in the range (0.0, 1.0),
an int in the range [1, inf) or None. Got -2.7755575615628914e-17 instead.
```

This occurred when `test_ratio` was effectively zero due to floating-point arithmetic errors (e.g., `train_ratio=0.9, val_ratio=0.1, test_ratio=0.0` could result in `test_ratio=-2.78e-17` after normalization).

## Root Cause

1. **Floating-point precision errors**: When ratios are normalized to sum to 1.0, small negative values can appear due to floating-point arithmetic
2. **Invalid sklearn parameter**: `train_test_split` requires `test_size` to be strictly in `(0.0, 1.0)` range - it doesn't accept negative values or values outside this range
3. **Missing zero-split handling**: The code didn't properly handle cases where test or val splits should be empty (ratio = 0)

## Solution

Applied comprehensive validation and error handling to all three dataset classes:

### Key Changes

1. **Ratio Validation & Clamping**
   ```python
   epsilon = 1e-6  # Tolerance for floating point comparison

   # Clamp ratios to valid range [0.0, 1.0]
   test_ratio_clean = max(0.0, min(1.0, self.test_ratio))
   val_ratio_clean = max(0.0, min(1.0, self.val_ratio))
   train_ratio_clean = max(0.0, min(1.0, self.train_ratio))

   # Determine if split should exist
   has_test_split = test_ratio_clean >= epsilon
   has_val_split = val_ratio_clean >= epsilon
   ```

2. **Conditional Split Creation**
   - Only call `train_test_split` if ratio >= epsilon
   - Return empty list `[]` for splits with ratio = 0
   - Handle edge cases in both stratified and non-stratified modes

3. **Improved Documentation**
   - Added clear notes about which splits are required/optional
   - train: always present
   - val: optional (can be 0)
   - test: optional (can be 0)

## Files Modified

### 1. ✅ [system/datautils/dataset_esc50.py](system/datautils/dataset_esc50.py)
- Modified `_apply_split()` method (lines 505-653)
- Added ratio validation and epsilon-based comparison
- Fixed both stratified and non-stratified splitting logic

### 2. ✅ [system/datautils/dataset_vegas.py](system/datautils/dataset_vegas.py)
- Modified `_apply_split()` method (lines 532-680)
- Applied identical fixes as ESC50
- Maintains compatibility with video_id sorting

### 3. ✅ [system/datautils/dataset_vggsound.py](system/datautils/dataset_vggsound.py)
- Modified `_apply_split()` method (lines 515-659)
- Applied identical fixes as ESC50 and VEGAS
- Ensures VGGSound has same robustness

## Behavior Changes

### Before Fix

```python
# Configuration with test_ratio = 0
dataset = ESC50Dataset(
    train_ratio=0.9,
    val_ratio=0.1,
    test_ratio=0.0  # Could become -2.78e-17 internally
)
# ❌ Raises: InvalidParameterError
```

### After Fix

```python
# Configuration with test_ratio = 0
dataset = ESC50Dataset(
    train_ratio=0.9,
    val_ratio=0.1,
    test_ratio=0.0
)
# ✅ Works correctly
# - train split: 90% of data
# - val split: 10% of data
# - test split: empty list []
```

### Supported Split Configurations

| train_ratio | val_ratio | test_ratio | Result |
|-------------|-----------|------------|--------|
| 0.7 | 0.1 | 0.2 | ✅ Standard 70-10-20 split |
| 0.8 | 0.2 | 0.0 | ✅ Train+Val only, no test |
| 0.9 | 0.0 | 0.1 | ✅ Train+Test only, no val |
| 1.0 | 0.0 | 0.0 | ✅ All data in train |
| 0.5 | 0.5 | 0.0 | ✅ Equal train/val, no test |

## Technical Details

### Epsilon Tolerance

```python
epsilon = 1e-6  # 0.000001
```

This tolerance handles floating-point precision errors:
- Ratios >= epsilon are treated as valid splits
- Ratios < epsilon are treated as zero (no split)

### Stratified Split Logic

For stratified splits using `sklearn.train_test_split`:

1. **First split**: train+val vs test (if test_ratio > epsilon)
2. **Second split**: train vs val (if val_ratio > epsilon)
3. **Relative size calculation**:
   ```python
   val_size_relative = val_ratio_clean / (train_ratio_clean + val_ratio_clean)
   # Ensures val_size_relative is in valid range (epsilon, 1-epsilon)
   ```

### Non-Stratified Split Logic

For manual splits without stratification:

1. **Group by class** for balanced splitting
2. **Calculate split sizes** using cleaned ratios
3. **Conditional extension**:
   - Only add samples to test if `has_test_split`
   - Only add samples to val if `has_val_split`
   - Train always gets samples

## Testing

All three datasets now handle:

✅ Standard splits (70-10-20)
✅ No test split (90-10-0)
✅ No val split (80-0-20)
✅ Only train (100-0-0)
✅ Floating-point edge cases
✅ Both stratified and non-stratified modes

## Backward Compatibility

The fix is **fully backward compatible**:
- Valid configurations continue to work as before
- Invalid configurations now work instead of raising errors
- No API changes required

## Usage Examples

### Example 1: Training Only (No Val/Test)

```python
dataset = ESC50Dataset(
    split=None,
    train_ratio=1.0,
    val_ratio=0.0,
    test_ratio=0.0,
    stratify=True
)

print(len(dataset.train))  # All samples
print(len(dataset.val))    # 0 (empty)
print(len(dataset.test))   # 0 (empty)
```

### Example 2: Train + Val Only

```python
dataset = VEGASDataset(
    split=None,
    train_ratio=0.9,
    val_ratio=0.1,
    test_ratio=0.0,  # No test split
    stratify=True
)

print(len(dataset.train))  # 90% of data
print(len(dataset.val))    # 10% of data
print(len(dataset.test))   # 0 (empty)
```

### Example 3: Standard Three-Way Split

```python
dataset = VGGSoundDataset(
    split=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    stratify=True
)

print(len(dataset.train))  # 70% of data
print(len(dataset.val))    # 10% of data
print(len(dataset.test))   # 20% of data
```

## Benefits

1. **Robustness**: Handles floating-point errors gracefully
2. **Flexibility**: Supports optional val/test splits
3. **Clarity**: Clear documentation of split requirements
4. **Consistency**: Same fix applied across all three datasets
5. **Safety**: Validates and clamps all ratios to valid ranges

## Related Issues

This fix resolves issues when:
- Using configs with `test_ratio=0` or `val_ratio=0`
- Ratios don't sum exactly to 1.0 due to floating-point precision
- Training adapters without test data (common use case)
- Using single-split configurations for quick experiments

---

**Last Updated**: 2026-01-10
**Applied To**: ESC50Dataset, VEGASDataset, VGGSoundDataset
**Status**: ✅ Complete and Tested
