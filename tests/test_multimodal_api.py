import pytest
from fastapi.testclient import TestClient

import pytest
try:
    from main_api import app as main_app
    from advanced_features.multimodal_ai import MultimodalAI
except Exception:
    main_app = None
    MultimodalAI = None


if main_app:
    client = TestClient(main_app)
else:
    client = None

@pytest.mark.skipif(not main_app or not client or not MultimodalAI or not getattr(MultimodalAI(), "transformers_ok", True), reason="transformers pipeline unavailable or app import failed")
def test_multimodal_infer_text():
    resp = client.post("/advanced/multimodal/infer", data={"text": "I love this AI!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "fusion" in data
    assert "text" in data or "text_sentiment" in data
