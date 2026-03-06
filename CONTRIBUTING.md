# Contributing to OGPR Radar Viewer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing](#testing)

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue!
- **Feature Requests**: Have an idea? We'd love to hear it!
- **Code**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve or translate documentation
- **Examples**: Share processing workflows or use cases
- **Testing**: Help test on different platforms or datasets

### Before You Start

1. Check existing issues to avoid duplicates
2. For large changes, open an issue first to discuss
3. Make sure you can run the application successfully

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ogpr-radar-viewer.git
cd ogpr-radar-viewer
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Install in editable mode
pip install -e .
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

---

## Contribution Guidelines

### Bug Reports

When filing a bug report, include:

- **Clear title**: Descriptive summary of the issue
- **Description**: What happened vs. what you expected
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Environment**: OS, Python version, package versions
- **Error messages**: Full error traceback if applicable
- **Sample data**: If possible, minimal OGPR file that reproduces issue

**Template:**

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
1. Load file 'example.ogpr'
2. Apply bandpass filter (100-800 MHz)
3. Click export
4. See error

**Expected Behavior**
Image should export successfully.

**Environment**
- OS: Windows 11
- Python: 3.10.5
- ogpr-viewer: 1.0.0
- PyQt6: 6.4.0

**Error Message**
```
Traceback (most recent call last):
  ...
```

**Additional Context**
File size: 500 MB, channels: 11
```

### Feature Requests

When requesting a feature:

- **Use case**: Explain why this feature would be useful
- **Description**: Clear description of desired functionality
- **Examples**: Show how it would work (mockups, pseudocode)
- **Alternatives**: Have you considered alternatives?

---

## Pull Request Process

### 1. Make Your Changes

- Write clear, documented code
- Follow coding standards (see below)
- Add tests if applicable
- Update documentation

### 2. Test Your Changes

```bash
# Run tests (when available)
pytest tests/

# Check code style
black ogpr_viewer/
flake8 ogpr_viewer/

# Test imports
python -c "from ogpr_viewer import OGPRParser, SignalProcessor"

# Test GUI (manual)
python -m ogpr_viewer.main
```

### 3. Commit Changes

```bash
git add .
git commit -m "Type: Brief description

Detailed explanation of changes.
Why this change was necessary."
```

**Commit message types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub:

- **Title**: Clear, descriptive title
- **Description**: What, why, and how
- **Related Issues**: Link to related issues
- **Screenshots**: If UI changes
- **Testing**: Describe testing performed

### 5. Review Process

- Maintainer will review your PR
- Address any requested changes
- Once approved, PR will be merged

---

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where helpful

**Formatting:**

```bash
# Auto-format with black
black ogpr_viewer/

# Check with flake8
flake8 ogpr_viewer/ --max-line-length=100
```

### Documentation

**Docstrings:** Use Google-style docstrings

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain purpose,
    algorithm, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
    
    Example:
        >>> function_name(42, "hello")
        True
    """
    pass
```

**Comments:**

```python
# Good: Explain WHY, not WHAT
# Use median instead of mean for robustness to outliers
background = np.median(data, axis=1)

# Bad: Obvious comment
# Calculate median
background = np.median(data, axis=1)
```

### Code Organization

**Imports:**

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from scipy import signal

# Local
from .ogpr_parser import OGPRParser
```

**Class Structure:**

```python
class MyClass:
    """Class docstring."""
    
    # Class variables
    CLASS_CONSTANT = 42
    
    def __init__(self):
        """Initialize."""
        # Instance variables
        self.data = None
    
    def public_method(self):
        """Public method."""
        pass
    
    def _private_method(self):
        """Private method (internal use)."""
        pass
```

---

## Testing

### Writing Tests

Tests will be added in `tests/` directory:

```python
# tests/test_parser.py
import pytest
from ogpr_viewer import OGPRParser

def test_parser_initialization():
    """Test parser can be initialized."""
    parser = OGPRParser('test_data.ogpr')
    assert parser is not None

def test_invalid_file():
    """Test parser raises error for invalid file."""
    with pytest.raises(FileNotFoundError):
        OGPRParser('nonexistent.ogpr')
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=ogpr_viewer --cov-report=html

# Specific test file
pytest tests/test_parser.py

# Specific test function
pytest tests/test_parser.py::test_parser_initialization
```

---

## Specific Contribution Areas

### Adding New Processing Algorithms

1. Add method to `SignalProcessor` class in `signal_processing.py`
2. Follow existing method structure (docstring, parameters, returns)
3. Add control widget in `main.py` if GUI needed
4. Document in user guide
5. Add example usage

### Adding New Visualization Options

1. Extend `RadarCanvas` class in `visualization.py`
2. Add UI control in `main.py`
3. Update display options group
4. Add to user guide

### Improving Performance

- Profile code to identify bottlenecks
- Consider numba JIT compilation for hot loops
- Implement parallel processing for batch operations
- Optimize memory usage for large files

---

## Questions?

Not sure about something?

- Open an issue asking for clarification
- Email: [your-email@example.com]
- Check existing issues and PRs

---

**Thank you for contributing! 🚀**