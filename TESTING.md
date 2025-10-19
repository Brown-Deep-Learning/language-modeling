# Testing Your HW5 Implementation

A comprehensive test suite has been provided in the `tests/` directory to help you verify your TODO implementations.

## Quick Start

### Install pytest (if needed)
```bash
pip install pytest
```

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run tests for a specific module
```bash
# Data processing
python tests/run_tests.py data

# RNNs
python tests/run_tests.py rnns

# Transformer
python tests/run_tests.py transformer

# Training
python tests/run_tests.py training

# Text generation
python tests/run_tests.py generation
```

## What's Included

### Test Files (5 modules)
- `test_data.py` - Data processing (TextTokenizer, sequences, datasets)
- `test_rnns.py` - RNN and LSTM models
- `test_transformer.py` - Transformer components and full model
- `test_training.py` - Training loop and checkpointing
- `test_text_generation.py` - Sampling and text generation

### Documentation
- `tests/README.md` - Full testing guide
- `tests/QUICK_REFERENCE.md` - Quick command reference
- `tests/TEST_SUITE_SUMMARY.md` - Detailed test coverage info

### Utilities
- `tests/run_tests.py` - Convenient test runner script
- `pytest.ini` - Pytest configuration

## Important Notes

⚠️ **These tests check BASIC functionality only!**

### What the tests DO check:
- ✓ Can your code run without errors?
- ✓ Are output shapes correct?
- ✓ Are data types correct?
- ✓ Are required methods/attributes present?

### What the tests DON'T check:
- ✗ Numerical correctness in detail
- ✗ Training quality/convergence
- ✗ Edge cases
- ✗ Performance
- ✗ Generated text quality

**Passing these tests does NOT guarantee full credit!** The autograder will have more rigorous tests.

## Recommended Workflow

1. **Start with data tests**
   ```bash
   python -m pytest tests/test_data.py -v
   ```

2. **Implement and test a model** (choose one first)
   ```bash
   python -m pytest tests/test_rnns.py -v
   # OR
   python -m pytest tests/test_transformer.py -v
   ```

3. **Test training**
   ```bash
   python -m pytest tests/test_training.py -v
   ```

4. **Test generation**
   ```bash
   python -m pytest tests/test_text_generation.py -v
   ```

5. **Final check**
   ```bash
   python -m pytest tests/ -v
   ```

## Getting Help

- Read `tests/README.md` for comprehensive documentation
- Check `tests/QUICK_REFERENCE.md` for command examples
- Look at the test code to see what's expected
- Ask on Piazza or come to office hours

## Remember

These tests are a **development aid**, not a **guarantee of correctness**. Always:
- Test on real data
- Verify training works
- Check generated text quality
- Review your code for correctness

Good luck! 🕵️
