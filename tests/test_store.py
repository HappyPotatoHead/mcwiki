import numpy as np
import pytest

from src import store


@pytest.fixture(autouse=True)
def tmp_index(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    yield
    store.reset()


def _chunks():
    return [
        {
            "id": "1:0",
            "title": "Alpha",
            "section": "(intro)",
            "url": "https://minecraft.wiki/wiki/Alpha",
            "pageid": 1,
            "content": "alpha body",
            "embed_text": "Alpha\n\nalpha body",
        },
        {
            "id": "2:0",
            "title": "Beta",
            "section": "(intro)",
            "url": "https://minecraft.wiki/wiki/Beta",
            "pageid": 2,
            "content": "beta body",
            "embed_text": "Beta\n\nbeta body",
        },
    ]


def test_upsert_query_roundtrip():
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.upsert(_chunks(), embeddings)

    assert store.count() == 2
    assert store.indexed_titles() == {"Alpha", "Beta"}

    hits = store.query(np.array([1.0, 0.0, 0.0]), top_n=2)
    assert hits[0]["title"] == "Alpha"
    assert hits[0]["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert hits[1]["title"] == "Beta"
    assert hits[1]["cosine"] == pytest.approx(0.0, abs=1e-5)


def test_delete_pages_and_reset():
    store.upsert(_chunks(), np.array([[1.0, 0.0], [0.0, 1.0]]))
    store.delete_pages([1])
    assert store.count() == 1
    assert store.indexed_titles() == {"Beta"}

    store.reset()
    assert store.count() == 0
