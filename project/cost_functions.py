import numpy as np
import cvxpy as cp
from abc import ABC


class SeparableCost(ABC):
    def __call__(self, x, z):
        pass

    def cost1(self, x):
        pass

    def cost2(self, x):
        pass


class MixedCost:
    def __init__(self, a, epsilon):
        self.a = a
        self.epsilon = epsilon

    def __call__(self, x, z):
        val = (1-self.epsilon) * self.a.T @ (z-x) + self.epsilon * np.sum((z-x)**2)
        return max(val, 0)

    def min_cost_positive(self, model, x, tol=0.000001):
        z = cp.Variable(len(x))
        function = cp.Minimize(
            cp.maximum((1-self.epsilon) * self.a.T @ (z-x) + self.epsilon * cp.sum((z-x)**2), 0)
        )
        constraints = [z @ model.coef_[0] >= -model.intercept_ + tol]
        prob = cp.Problem(function, constraints)
        prob.solve()
        cost = cp.maximum((1-self.epsilon) * self.a.T @ (z-x) + self.epsilon * cp.sum((z-x)**2), 0)
        return z, cost

    def maximize_features_no_cost(self, model, x, tol=0.000001):
        x_prime, change_cost = self.min_cost_positive(model, x, tol)
        if model.predict(x_prime.value.reshape(1, -1))[0] == 1:
            return x_prime.value
        else:
            return x_prime.value + 2*tol

    def maximize_features(self, model, x, tol=0.000001):
        x_prime, change_cost = self.min_cost_positive(model, x, tol)
        if (change_cost.value < 2) and (model.predict(x_prime.value.reshape(1, -1))[0] == 1):
            return x_prime.value
        else:
            return x


class LinearCost(SeparableCost):
    def __init__(self, a):
        self.a = a

    def cost1(self, x):
        return self.a.T @ x

    def cost2(self, z):
        return self.a.T @ z

    def __call__(self, x, z):
        val = self.cost2(z) - self.cost1(x)
        return max(val, 0)
