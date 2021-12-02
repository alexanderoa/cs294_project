import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import LinearSVC

'''
In utils.py, maybe should find a way to just import them
'''
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

def best_respond(model, x, cost):
    if model.predict(x.reshape(1, -1))[0] == 1:
        return x
    return cost.maximize_features(model, x)

def generate_friend_dict(df, m, featurelist = None, model = None):
    if model is not None:
        preds = model.predict(df[featurelist])
        neg = preds[preds == -1].index.values
        pos = preds[preds == 1].index.values
    else:
        neg = df[df['target'] == -1].index.values
        pos = df[df['target'] == 1].index.values
    friends = dict()

    for i in range(len(df)):
        num_negative = np.random.randint(1, (m-2), size=1) #at least one person who is positive
        num_positive = m - num_negative
        neg_idx = np.random.choice(neg, num_negative)
        pos_idx = np.random.choice(pos, num_positive)
        friends[i] = np.union1d(neg_idx, pos_idx)
    return friends

def generate_random_friends_dict(df, m):
    friends = dict()
    for i in range(len(df)):
        friends[i] = np.random.randint(low = 0, high=len(df)-1, size=m)
    return friends

def generate_new_friends_dict(df, m, d, featurelist = None, model = None):
    new_df = create_dataset(data_size=m, d=d)
    friends = dict()
    if model is not None:
        preds = model.predict(new_df[featurelist])
        neg = preds[preds == -1].index.values
        pos = preds[preds == 1].index.values
    else:
        neg = new_df[new_df['target'] == -1].index.values
        pos = new_df[new_df['target'] == 1].index.values

    for i in range(len(df)):
        num_negative = np.random.randint(1, (m-2), size=1) #at least one person who is positive
        num_positive = m - num_negative
        neg_idx = np.random.choice(neg, num_negative)
        pos_idx = np.random.choice(pos, num_positive)
        friends[i] = np.union1d(neg_idx, pos_idx)
    return friends


def strategy_light(original_df, model, featurelist, cost, players = None, keys = None):
    '''
    players: list of indices for players that act strategically
    keys: list of MemberKeys for players that act strategically
    '''
    if players is None and keys is None:
        print('Must initialize either players or keys')
        exit()
    if players is None:
        players = original_df[original_df['MemberKey'].isin(keys)].index.values

    modify_df = original_df[featurelist].copy()
    with tqdm(total=len(players)) as t:
        for p in players:
            x = np.array(modify_df.loc[p])
            x_br = best_respond(model, x, cost)
            modify_df.loc[p] = x_br
            t.update(1)
    for col_name in filter(lambda c: c not in modify_df.columns, original_df.columns):
        modify_df.insert(len(modify_df.columns), col_name, original_df[col_name], True)

    return modify_df

def strategy_dark(original_df, model, featurelist, cost, friends_dict, players = None, keys = None):
    '''
    players: list of indices for players that act strategically
    keys: list of MemberKeys for players that act strategically
    '''
    if players is None and keys is None:
        print('Must initialize either players or keys')
        exit()
    if players is None:
        players = original_df[original_df['MemberKey'].isin(keys)].index.values
    modify_df = original_df[featurelist].copy()
    fhat_dict = dict()
    train_dict = dict()
    with tqdm(total=len(players)) as t:
        for p in players:
            train = original_df.loc[friends_dict[p]]
            f_hat = LinearSVC(max_iter=10000)
            f_hat.fit(train[featurelist], model.predict(train[featurelist]))
            x = np.array(modify_df.loc[p])
            x_br = best_respond(f_hat, x, cost)
            modify_df.loc[p] = x_br
            fhat_dict[p] = f_hat
            train_dict[p] = train
            t.update(1)
    for col_name in filter(lambda c: c not in modify_df.columns, original_df.columns):
        modify_df.insert(len(modify_df.columns), col_name, original_df[col_name], True)

    return modify_df, train_dict, fhat_dict

def strategy_with_explanations(original_df, model, featurelist, cost, friends_dict, players = None, keys = None):
    #results seem a bit strange, I should double check what's going on
    '''
    m now relates to the number of people who have applied before you and received counterfactual explanations
    '''
    if players is None and keys is None:
        print('Mus initialize either players or keys')
        exit()
    if players is None:
        players = original_df[original_df['MemberKey'].isin(keys)].index.values
    modify_df = original_df[featurelist].copy()
    fhat_dict = dict()
    train_dict = dict()
    with tqdm(total=len(players)) as t:
        for p in players:
            explain_df = pd.DataFrame()
            for i in friends_dict[p]:
                x = np.array(original_df[featurelist].loc[i])
                x_t = cost.maximize_features_no_cost(model, x)
                x_df = from_numpy_to_panda_df(x_t.reshape(1, -1))
                x_df.insert(0, 'MemberKey', original_df['MemberKey'].loc[i])
                x_df.insert(0, 'target', model.predict(x_t.reshape(1, -1)))
                explain_df = explain_df.append(x_df)
            train = pd.concat([original_df.loc[friends_dict[p]], explain_df])
            f_hat = LinearSVC(max_iter=10000)
            f_hat.fit(train[featurelist], model.predict(train[featurelist]))
            x = np.array(modify_df.loc[p])
            x_br = best_respond(f_hat, x, cost)
            modify_df.loc[p] = x_br
            fhat_dict[p] = f_hat
            train_dict[p] = train
            t.update(1)
    return modify_df, train_dict, fhat_dict
