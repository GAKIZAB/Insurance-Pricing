import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from src.models.base_model import PricingModel

class GLMPricingModel(PricingModel):
    def __init__(self, name="GLM", formula=None):
        super().__init__(name)
        self.formula = formula

    def train(self, data, target='ClaimNb', exposure='Exposure'):
        self.model = smf.glm(formula=self.formula, data=data, family=sm.families.Poisson(), offset=np.log(data[exposure])).fit()
        return self.model

    def predict(self, data):
        return self.model.predict(data)
        
    def get_frequency(self, data, exposure_col='Exposure'):
        return self.model.predict(data) / data[exposure_col]
