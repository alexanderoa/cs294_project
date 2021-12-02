import numpy as np
from tqdm import tqdm


class HardtModel:
    def __init__(self, separable_cost):
        self.separable_cost = separable_cost
        self.coef_ = (separable_cost.a, None)
        self.intercept_ = None
        self.min_si = None

    def __call__(self, X):
        def single_prediction(x):
            return 1 if self.separable_cost.cost2(x) >= self.min_si else -1

        if self.min_si is None:
            print('Model not trained yet...')
            return

        if isinstance(X, np.ndarray):
            return np.apply_along_axis(single_prediction, 1, X)
        else:
            return X.apply(single_prediction, axis=1)

    def predict(self, X):
        return self(X)

    def fit(self, X, y):
        assert(len(X.iloc[0]) == len(self.separable_cost.a)), 'Data and weight vector have different dimensions'
        min_err_si = np.inf
        thresh = min_err_si

        def threshold_func(x):
            return 1 if self.separable_cost.cost1(x) >= thresh else -1
        S = X.apply(self.separable_cost.cost2, axis=1) + 2

        with tqdm(total=len(S)) as t:
            for i, s_i in enumerate(S):
                thresh = s_i - 2
                err_si = np.sum(y != X.apply(threshold_func, axis=1))
                if err_si < min_err_si:
                    min_err_si = err_si
                    self.min_si = s_i
                t.update(1)
        self.intercept_ = -self.min_si

