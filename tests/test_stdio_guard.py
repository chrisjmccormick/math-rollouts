"""The notebook-kernel stdio guard around vLLM engine init: ipykernel's
sys.stdout/stderr raise ``io.UnsupportedOperation`` on ``fileno()``, which vLLM's
``suppress_stdout`` calls during engine-core setup (the classic Colab crash)."""
from __future__ import annotations

import io
import sys

from math_rollouts.generate.natural import _filenoable_stdio


class _KernelStream(io.StringIO):
    """Mimics ipykernel's OutStream: writable, but fileno() is unsupported."""

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


def test_broken_stdout_swapped_and_restored():
    old_out, old_err = sys.stdout, sys.stderr
    fake = _KernelStream()
    sys.stdout = fake
    try:
        with _filenoable_stdio():
            sys.stdout.fileno()                    # what vLLM init needs to work
            assert sys.stdout is not fake
            if old_err is sys.stderr:              # healthy stderr left alone
                assert sys.stderr is old_err
        assert sys.stdout is fake                  # restored after the block
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_healthy_streams_untouched():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout.fileno()
        sys.stderr.fileno()
    except Exception:
        import pytest
        pytest.skip("test runner's own streams lack fileno()")
    with _filenoable_stdio():
        assert sys.stdout is old_out and sys.stderr is old_err
