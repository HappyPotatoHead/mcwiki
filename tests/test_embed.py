import numpy as np

from src.embed import cosine_similarity


def test_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == 1.0


def test_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-9
