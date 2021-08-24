import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score


class LinearRegression(linear_model.LinearRegression):
    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self).__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / se
        self.t = np.round(self.t, 3)
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        self.p = np.round(self.p[0], 3)
        return self

def get_correlation(x, y):
    return np.round(stats.pearsonr(x, y)[0], 3)


# tests
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
reg = LinearRegression()
reg.fit(x,y)
r_sq = np.round(reg.score(x, y), 3)

print(f"r_sq: {r_sq}")
print(f"reg.p: {reg.p}")
print(f"corr(x1,x2): {get_correlation([a[0] for a in x], [a[1] for a in x])}")
