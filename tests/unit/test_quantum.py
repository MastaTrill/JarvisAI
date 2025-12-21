"""
Unit test for quantum processor
"""
import pytest
from quantum_processor import QuantumProcessor

def test_create_superposition():
    qp = QuantumProcessor(creator_auth=True)
    result = qp.create_quantum_superposition(num_states=4)
    assert isinstance(result, dict)
    assert 'probabilities' in result
    assert len(result['probabilities']) == 4
    assert abs(sum(result['probabilities']) - 1.0) < 1e-6
