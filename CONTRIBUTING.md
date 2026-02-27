# Contributing to QNHD-Fold

Thank you for your interest in contributing to QNHD-Fold! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

- Use GitHub Issues to report bugs
- Include detailed steps to reproduce
- Provide system information (OS, Python version, etc.)
- Include error messages and stack traces

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain the motivation and use cases

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest tests/`)
6. Format code with black (`black qnhd_fold/`)
7. Commit with descriptive message
8. Push to your fork
9. Open a Pull Request

## Code Style

- Follow PEP 8
- Use black for formatting
- Add docstrings to all functions
- Include type hints where possible

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py

# With coverage
pytest --cov=qnhd_fold tests/
```

## Questions?

Contact: tmarena@cua.edu or open a GitHub Discussion
