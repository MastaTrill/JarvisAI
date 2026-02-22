import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from analyze_generated_images import analyze_image, perform_clustering_and_pca


def test_analyze_image_grayscale():
    arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    img = Image.fromarray(arr)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        stats = analyze_image(tmp.name)
        assert stats["mean"] == pytest.approx(np.mean(arr), rel=1e-2)
        assert "hist_sample" in stats
    os.remove(tmp.name)


def test_analyze_image_color():
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        stats = analyze_image(tmp.name)
        assert "mean_R" in stats and "mean_G" in stats and "mean_B" in stats
        assert "colorfulness" in stats
    os.remove(tmp.name)


def test_perform_clustering_and_pca():
    df = pd.DataFrame(
        {
            "mean": np.random.rand(10),
            "std": np.random.rand(10),
            "unique_values": np.random.randint(1, 255, 10),
            "hist_entropy": np.random.rand(10),
        }
    )
    out_dir = tempfile.mkdtemp()
    df2 = perform_clustering_and_pca(
        df, ["mean", "std", "unique_values", "hist_entropy"], out_dir
    )
    assert "cluster" in df2.columns
    assert "pca1" in df2.columns and "pca2" in df2.columns
