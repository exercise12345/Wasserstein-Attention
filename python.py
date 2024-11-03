import torch
import numpy as np
from scipy.optimize import linprog
from WassersteinSelfAttentionModel import WassersteinSelfAttentionModel

class WassersteinSelfAttentionCaller:
    def __init__(self):
        self.model = None

    def create_model(self, in_dim, hidden_dim):
        self.model = WassersteinSelfAttentionModel(in_dim, hidden_dim)

    def run_model(self, input_data):
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
        return self.model(input_data)

    def test(self):
        in_dim = 64
        hidden_dim = 32
        self.create_model(in_dim, hidden_dim)

        batch_size = 16
        seq_len = 10
        x = torch.randn(batch_size, seq_len, in_dim)
        output = self.run_model(x)
        print(f"Output shape: {output.shape}")
        print(output)



if __name__ == "__main__":
    caller = WassersteinSelfAttentionCaller()
    caller.test()
