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


    