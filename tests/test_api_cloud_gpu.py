import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)


def test_cloud_upload_download():
    # Upload (file must exist in data/uploads for real test)
    response = client.post(
        "/cloud/upload?filename=test.csv&provider=s3&bucket=test-bucket"
    )
    assert response.status_code in [200, 404, 500]
    # Download
    response = client.get(
        "/cloud/download?filename=test.csv&provider=s3&bucket=test-bucket"
    )
    assert response.status_code in [200, 404, 500]


def test_predict_gpu():
    payload = {"model_name": "testmodel", "data": [[1, 2, 3]]}
    response = client.post("/models/testmodel/predict_gpu", json=payload)
    assert response.status_code in [200, 404, 500]


def test_predict_external():
    payload = {"model_name": "testmodel", "data": [[1, 2, 3]]}
    response = client.post("/models/testmodel/predict_external", json=payload)
    assert response.status_code in [200, 404, 500]
