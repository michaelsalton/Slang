import numpy as np

from slang_research import load_module


def test_add():
    module = load_module("hello.slang")
    a = np.arange(16, dtype=np.float32)
    b = np.ones(16, dtype=np.float32)
    np.testing.assert_allclose(module.add(a, b, _result="numpy"), a + b)


def test_polynomial():
    module = load_module("hello.slang")
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    result = module.polynomial(2.0, 8.0, -1.0, x, _result="numpy")
    np.testing.assert_allclose(result, 2 * x**2 + 8 * x - 1, rtol=1e-5)
