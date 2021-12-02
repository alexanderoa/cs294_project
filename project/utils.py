import pandas as pd
import numpy as np

#straight from strategic_rep repo
def from_numpy_to_panda_df(data):
    data = pd.DataFrame(data=data[0:, 0:], index=[i for i in range(data.shape[0])],
                        columns=['f' + str(i) for i in range(data.shape[1])])
    return data


def create_dataset(data_size, covariance=None, d=1):
    def map_sum_one_minus_one(sum_value):
        return 1 if sum_value >= 0 else -1

    if covariance is None:
        covariance = np.eye(d)
    means = np.zeros(shape=d)
    data = np.random.multivariate_normal(mean=means, cov=covariance, size=data_size)
    data = from_numpy_to_panda_df(data)
    # memberkeys:
    member_keys = [f's{i}' for i in range(data_size)]
    data.insert(len(data.columns), 'MemberKey', member_keys, allow_duplicates=True)
    labels = list(data.sum(axis=1).apply(map_sum_one_minus_one))
    data.insert(len(data.columns), 'target', labels, allow_duplicates=True)
    return data
