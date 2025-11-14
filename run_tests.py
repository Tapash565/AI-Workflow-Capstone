#!/usr/bin/env python
"""Run the full test suite (wrapper around pytest).

Usage:
  python run_tests.py
"""
import sys
import pytest

if __name__ == '__main__':
    errno = pytest.main(['-q'])
    sys.exit(errno)
