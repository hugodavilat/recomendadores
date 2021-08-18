#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class FeaturesCalculator():
    def __init__(self, urm, uir):
        self.urm = urm
        self.uir = uir
        (self.U, self.I) = self.urm.get_shape()
        self.R = self.urm.count_nonzero()
    
    def space_size(self):
        return self.U*self.I
    
    def shape(self):
        return self.U/self.I
    
    def density(self):
        return self.R/(self.U*self.I)
    
    def rpu(self):
        return self.R/self.U

    def rpi(self):
        return self.R/self.I
    
    def _gini(self, x):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g
    
    def gini_i(self):
        x = []
        for j in range(self.I):
            col = self.urm.getcol(j)
            x.append(col.count_nonzero())
        return self._gini(x)
    
    def gini_u(self):
        x = []
        for i in range(self.U):
            row = self.urm.getrow(i)
            x.append(row.count_nonzero())
        return self._gini(x)
    


urm = pickle.load(open("../datasets/ml-100k/URM/001_URM.pk", 'rb'))
df = pd.read_csv("../datasets/ml-100k/UIR/001_UIR.csv")
uir = [tuple(row) for row in df.values]
c = FeaturesCalculator(urm, uir)

print(c.space_size(), c.shape(), c.density(), c.rpu(), c.rpi(), c.gini_u(), c.gini_i())