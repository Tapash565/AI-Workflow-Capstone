class DummyModel:
    def predict(self, X):
        import numpy as np
        try:
            n = X.shape[0]
        except Exception:
            n = 1
        return np.zeros(n)
