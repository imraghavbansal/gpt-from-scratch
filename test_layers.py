import numpy as np

# from gpt import EmbeddingLayer, Softmax

# # 1. Softmax rows sum to 1
# sm = Softmax()
# x = np.random.randn(3, 5)
# out = sm.forward(x)
# print("row sums:", out.sum(axis=1))

# # 2. Shift invariance + stability with huge values
# print("stable:", np.allclose(sm.forward(x), sm.forward(x + 1000)))

# # 3. Gradient check for softmax backward
# W = np.random.randn(3, 5)           # fixed weights, so loss = sum(softmax(x) * W)
# sm.forward(x)
# analytic = sm.backward(W)           # dL/dout = W

# numeric = np.zeros_like(x)
# eps = 1e-5
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         xp, xm = x.copy(), x.copy()
#         xp[i, j] += eps
#         xm[i, j] -= eps
#         numeric[i, j] = (np.sum(sm.forward(xp) * W) - np.sum(sm.forward(xm) * W)) / (2 * eps)

# print("grad check:", np.allclose(analytic, numeric, atol=1e-6))

# # 4. Embedding: repeated token gradients add up
# emb = EmbeddingLayer(10, 4)
# ids = np.array([[1, 2, 1]])
# emb.forward(ids)
# emb.backward(np.ones((1, 3, 4)))
# print("token 1 grad (expect 2s):", emb.grad_embeddings[1])


from gpt import CrossEntropyWithSoftmax

print("--- cross entropy ---")
B, T, V = 2, 3, 5
ce = CrossEntropyWithSoftmax()
logits = np.random.randn(B, T, V)
labels = np.random.randint(0, V, size=(B, T))

# 1. Uniform logits -> loss should be log(V)
print("uniform loss ok:", np.isclose(ce.forward(np.zeros((B, T, V)), labels), np.log(V)))

# 2. Stability: huge logits must not give nan/inf
print("stable:", np.isfinite(ce.forward(logits * 1000, labels)))

# 3. Gradient check
ce.forward(logits, labels)
analytic = ce.backward()
numeric = np.zeros_like(logits)
eps = 1e-5
for idx in np.ndindex(*logits.shape):
    lp, lm = logits.copy(), logits.copy()
    lp[idx] += eps
    lm[idx] -= eps
    numeric[idx] = (ce.forward(lp, labels) - ce.forward(lm, labels)) / (2 * eps)
print("grad check:", np.allclose(analytic, numeric, atol=1e-6))