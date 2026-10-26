import numpy as np
from gpt import EmbeddingLayer, Softmax, CrossEntropyWithSoftmax

np.random.seed(0)  # reproducible tests
EPS = 1e-5


def numeric_grad(f, x):
    """Numerical gradient of scalar function f at x (central differences)."""
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += EPS
        xm[idx] -= EPS
        g[idx] = (f(xp) - f(xm)) / (2 * EPS)
    return g


def test_softmax():
    sm = Softmax()
    x = np.random.randn(3, 5)

    # rows are valid probabilities
    out = sm.forward(x)
    assert np.allclose(out.sum(axis=1), 1), "rows must sum to 1"
    assert np.all(out > 0), "probabilities must be positive"

    # shift invariance + stability with huge values
    assert np.allclose(sm.forward(x), sm.forward(x + 1000)), "must be shift invariant"
    assert np.all(np.isfinite(sm.forward(x * 1000))), "must not overflow"

    # gradient check: loss = sum(softmax(x) * W)
    W = np.random.randn(3, 5)
    sm.forward(x)
    analytic = sm.backward(W)
    numeric = numeric_grad(lambda z: np.sum(sm.forward(z) * W), x)
    assert np.allclose(analytic, numeric, atol=1e-6), "softmax grad check failed"
    print("softmax: ok")


def test_embedding():
    emb = EmbeddingLayer(10, 4)
    ids = np.array([[1, 2, 1]])

    out = emb.forward(ids)
    assert out.shape == (1, 3, 4), "output shape must be (batch, seq, dim)"
    assert np.allclose(out[0, 0], emb.embeddings[1]), "lookup must return the right row"

    # a token that appears twice gets its gradients added
    emb.backward(np.ones((1, 3, 4)))
    assert np.allclose(emb.grad_embeddings[1], 2), "repeated token grads must add up"
    assert np.allclose(emb.grad_embeddings[2], 1), "single token grad must be 1"
    assert np.allclose(emb.grad_embeddings[0], 0), "unused token grad must be 0"
    print("embedding: ok")


def test_cross_entropy():
    B, T, V = 2, 3, 5
    ce = CrossEntropyWithSoftmax()
    logits = np.random.randn(B, T, V)
    labels = np.random.randint(0, V, size=(B, T))

    # uniform logits -> loss = log(V)
    uniform = ce.forward(np.zeros((B, T, V)), labels)
    assert np.isclose(uniform, np.log(V)), "uniform logits -> loss = log(V)"

    # matches the naive softmax -> log -> pick formula on safe inputs
    flat = logits.reshape(B * T, V)
    e = np.exp(flat - flat.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    naive = -np.mean(np.log(probs[np.arange(B * T), labels.reshape(-1)]))
    assert np.isclose(ce.forward(logits, labels), naive), "must match naive cross-entropy"

    # stays finite with huge logits
    assert np.isfinite(ce.forward(logits * 1000, labels)), "must stay finite"

    # gradient check
    ce.forward(logits, labels)
    analytic = ce.backward()
    numeric = numeric_grad(lambda z: ce.forward(z, labels), logits)
    assert np.allclose(analytic, numeric, atol=1e-6), "cross-entropy grad check failed"
    print("cross-entropy: ok")


if __name__ == "__main__":
    test_softmax()
    test_embedding()
    test_cross_entropy()
    print("all tests passed")