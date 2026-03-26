"""
Unit test for quantum processor
"""

from src.quantum.quantum_processor import QuantumProcessor


def test_create_superposition():
    qp = QuantumProcessor()
    qp.authenticate_creator("AETHERON_QUANTUM_CREATOR_KEY_2025")
    result = qp.create_quantum_superposition(["state0", "state1", "state2", "state3"])
    assert isinstance(result, dict)
    assert result["status"] == "success"
    superposition = result["superposition"]
    assert superposition["total_states"] == 4
    assert abs(superposition["probability_per_state"] * 4 - 1.0) < 1e-6
