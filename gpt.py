import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.input = None

        # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, inputs):
        self.input = inputs
        return self.embeddings[inputs]

    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(grad_embeddings, self.input, prev_grad)

        self.grad_embeddings += grad_embeddings
        return grad_embeddings


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        #Softmax is shift invariant so we can subtract the max for numerical stability
        max_logit = np.max(inputs, axis=1, keepdims=True)

        # Softmax Ouptut
        softmax_out = np.exp(inputs - max_logit)/ np.sum(np.exp(inputs - max_logit), axis=1, keepdims=True)
        self.softmax_out = softmax_out
        return softmax_out

    def backward(self, prev_grad):
        return prev_grad * self.softmax_out - self.softmax_out * np.sum(self.softmax_out * prev_grad, axis=1, keepdims=True)


class CrossEntropyWithSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None
        self.B = self.T = self.V = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B * T, V))
        labels_flat = np.reshape(labels, (B * T))

        # Numerically stable cross entropy loss (log-sum-exp trick)
        max_logit = np.max(logits_flat, axis=1, keepdims=True)
        diff = logits_flat - max_logit

        lse = max_logit + np.log(np.sum(np.exp(diff), axis=1, keepdims=True))
        log_softmax = np.expand_dims(logits_flat[np.arange(B * T), labels_flat], axis=1) - lse

        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B, self.T, self.V = B, T, V

        return -np.sum(log_softmax) / logits_flat.shape[0]

    def backward(self):
        N = self.B * self.T
        softmax = np.exp(self.logits_flat - self.lse)        # (N, V)
        softmax[np.arange(N), self.labels_flat] -= 1         # subtract one-hot
        grad = softmax / N
        return grad.reshape(self.B, self.T, self.V)

    def get_name(self):
        return "CrossEntropyWithSoftmax"
