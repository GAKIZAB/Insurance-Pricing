from abc import ABC, abstractmethod

class PricingModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass
