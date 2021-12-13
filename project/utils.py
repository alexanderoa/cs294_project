import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_sim(plot_df, model):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    ax = plt.figure(figsize = (10,10))
    ax = sns.scatterplot(data=plot_df, x='f0', y='f1', hue='target', palette=sns.color_palette('tab10')[:2])
    for i in range(len(plot_df)):
        if plot_df['f0_after'].loc[i] - plot_df['f0'].loc[i] < 0.05:
            continue
        ax.arrow(x=plot_df['f0'].loc[i],
                 y=plot_df['f1'].loc[i],
                 dx=plot_df['f0_after'].loc[i] - plot_df['f0'].loc[i],
                 dy=plot_df['f1_after'].loc[i] - plot_df['f1'].loc[i],
                 width=0.005,
                 head_width=0.05,
                 head_length=0.05,
                 color='red',
                 alpha=0.5)
    t = np.arange(-2, 4, 0.2)
    plt.plot(t, -t - model.intercept_, zorder=0, label='Strategic')
    plt.plot(t, -t - model.intercept_ - 2, '--', zorder=0, label='Non-strategic')
    plt.xlim(-2, 2.5)
    plt.ylim(-2, 2.5)

def plot_sim_iter(plot_df, model, i, axes):
    sns.scatterplot(data=plot_df, x='f0', y='f1', hue='target',
                              palette=sns.color_palette('tab10')[:2], ax = axes[i])
    for i in range(len(plot_df)):
        if plot_df['f0_after'].loc[i] - plot_df['f0'].loc[i] < 0.05:
            continue
        axes[i].arrow(x=plot_df['f0'].loc[i],
                 y=plot_df['f1'].loc[i],
                 dx=plot_df['f0_after'].loc[i] - plot_df['f0'].loc[i],
                 dy=plot_df['f1_after'].loc[i] - plot_df['f1'].loc[i],
                 width=0.005,
                 head_width=0.05,
                 head_length=0.05,
                 color='red',
                 alpha=0.5)
    t = np.arange(-2, 4, 0.2)
    axes[i].plot(t, -t - model.intercept_, zorder=0)
    axes[i].plot(t, -t - model.intercept_ - 2, '--', zorder=0)
    axes[i].xlim(-2, 2)
    axes[i].ylim(-2, 2)
