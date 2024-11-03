import torch

class WassersteinSelfAttentionCaller:
    def __init__(self):
        self.model = None

    def create_model(self, in_dim, hidden_dim):
        self.model = WassersteinSelfAttentionModel(in_dim, hidden_dim)

    def run_model(self, input_data):
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")
        return self.model(input_data)
