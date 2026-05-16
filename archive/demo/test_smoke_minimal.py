"""Minimal smoke test (no conftest, no heavy imports)."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_quantum_processor_instantiation():
    """Test that QuantumProcessor can be instantiated."""
    from src.quantum.quantum_processor import QuantumProcessor

    qp = QuantumProcessor()
    assert qp is not None
    assert hasattr(qp, "authenticate_creator")
    assert qp.processor_id.startswith("QUANTUM_PROC_")


def test_quantum_processor_auth():
    """Test Creator authentication."""
    from src.quantum.quantum_processor import QuantumProcessor

    qp = QuantumProcessor()
    result = qp.authenticate_creator("AETHERON_QUANTUM_CREATOR_KEY_2025")
    assert result is True
    assert qp.creator_authorized is True


def test_python_environment():
    """Test Python and package availability."""
    import numpy
    import pytest

    assert numpy.__version__
    assert pytest.__version__
    assert sys.version_info.major == 3


if __name__ == "__main__":
    test_python_environment()
    test_quantum_processor_instantiation()
    test_quantum_processor_auth()
    print("✅ All smoke tests passed!")
