import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import d3rlpy
from sys import stdout
# from poison_detection import ActivationDefence

from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from mujoco_poisoned_dataset import poison_hopper, poison_walker2d, poison_half
import random
import argparse
from d3rlpy.algos import BC, BCQ, BEAR, CQL, AWAC, SAC, IQL, PLASWithPerturbation, TD3PlusBC

def hopper_activation_clustering():
    data = np.load('./detection/hidden_layer.npy')
    pca = PCA(n_components=3)
    low_den_data = pca.fit(data.T)
    result = KMeans(n_clusters=2).fit(low_den_data.components_.T)
    # visualization
    colors = ["#0000FF", "#00FF00"]
    colors_ = []
    for i in result.labels_:
        colors_.append(colors[i])

    num_1 = np.sum(result.labels_)
    print(num_1)
    #plt.scatter(low_den_data.components_.T[:, 0], low_den_data.components_.T[:, 1], color = colors_)
    #plt.savefig('./hopper_detect.png')


    # calcuate TurePostive, FalseNegative, TrueNegative, FalsePositive
    clean_dataset_label, poisoned_dataset_label = np.split(result.labels_, [1142502])
    if num_1 < 450000:
        true_clean_label = np.zeros(1142502),
        true_poisoned_label = np.ones(221868)
    elif num_1 > 1142502 + 221868 - 450000:
        true_clean_label = np.ones(1142502),
        true_poisoned_label = np.zeros(221868)
    else:
        print('Escape the backdoor detection!')
        return

    TurePostive = np.sum(np.where(poisoned_dataset_label - true_poisoned_label, 0, 1))
    TureNegative = np.sum(np.where(clean_dataset_label - true_clean_label, 0, 1))
    FalseNegative = 221868 - TurePostive
    FalsePostive = 1142502 - TureNegative

    print_f1_score(TurePostive, FalseNegative, TureNegative, FalsePostive)
    return

def half_activation_clustering():
    data = np.load('./detection/hidden_layer.npy')
    pca = PCA(n_components=3)
    low_den_data = pca.fit(data.T)
    result = KMeans(n_clusters=2).fit(low_den_data.components_.T)
    # visualization
    colors = ["#0000FF", "#00FF00"]
    colors_ = []
    for i in result.labels_:
        colors_.append(colors[i])

    num_1 = np.sum(result.labels_)
    print(num_1)
    #plt.scatter(low_den_data.components_.T[:, 0], low_den_data.components_.T[:, 1], color = colors_)
    #plt.savefig('./hopper_detect.png')


    # calcuate TurePostive, FalseNegative, TrueNegative, FalsePositive
    clean_dataset_label, poisoned_dataset_label = np.split(result.labels_, [899000])
    if num_1 < 350000:
        true_clean_label = np.zeros(899000),
        true_poisoned_label = np.ones(99800)
    elif num_1 > 899000 + 99800 - 350000:
        true_clean_label = np.ones(899000),
        true_poisoned_label = np.zeros(99800)
    else:
        print('Escape the backdoor detection!')
        return

    TurePostive = np.sum(np.where(poisoned_dataset_label - true_poisoned_label, 0, 1))
    TureNegative = np.sum(np.where(clean_dataset_label - true_clean_label, 0, 1))
    FalseNegative = 99800 - TurePostive
    FalsePostive = 899000 - TureNegative

    print_f1_score(TurePostive, FalseNegative, TureNegative, FalsePostive)
    return

def walker_activation_clustering():
    data = np.load('./detection/hidden_layer.npy')
    pca = PCA(n_components=3)
    low_den_data = pca.fit(data.T)
    result = KMeans(n_clusters=2).fit(low_den_data.components_.T)
    # visualization
    colors = ["#0000FF", "#00FF00"]
    colors_ = []
    for i in result.labels_:
        colors_.append(colors[i])

    num_1 = np.sum(result.labels_)
    print(num_1)
    #plt.scatter(low_den_data.components_.T[:, 0], low_den_data.components_.T[:, 1], color = colors_)
    #plt.savefig('./hopper_detect.png')


    # calcuate TurePostive, FalseNegative, TrueNegative, FalsePositive
    clean_dataset_label, poisoned_dataset_label = np.split(result.labels_, [897192])

    if num_1 < 350000:
        true_clean_label = np.zeros(897192),
        true_poisoned_label = np.ones(103511),
    elif num_1 > 897192 + 103511 - 350000:
        true_clean_label = np.ones(897192),
        true_poisoned_label = np.zeros(103511),
    else:
        print('Escape the backdoor detection!')
        return



    TurePostive = np.sum(np.where(poisoned_dataset_label - true_poisoned_label, 0, 1))
    TureNegative = np.sum(np.where(clean_dataset_label - true_clean_label, 0, 1))
    FalseNegative = 103511 - TurePostive
    FalsePostive = 897192 - TureNegative

    print_f1_score(TurePostive, FalseNegative, TureNegative, FalsePostive)

def print_f1_score(tp, fn, tn, fp):

    if tp + fp == 0 or tp + fn == 0:
        print('escape the detection')
        return

    # accuracy = float(tp + fp) / tp + fn + tn + fp
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    if tp == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    # print('accuracy = ', accuracy)
    print('precision = ', precision)
    print('recall = ', recall)
    print('f1 = ', f1)

def detect_hopper(args):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    poison_dataset = poison_hopper()
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=False)
    train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
                                                                   train_size= args.poison_rate,
                                                                   shuffle=False)
    # num_clean = 1142502
    num_clean = 0
    for i in train_episodes:
        for j in i.observations:
            num_clean +=1
    # num_poison = 221868
    num_poison = 0
    for i in train_poison_episodes:
        for j in i.observations:
            num_poison +=1

    train_episodes.extend(train_poison_episodes)
    cql = PLASWithPerturbation.from_json('./poisoned_model_traing/hopper_trigger_plaswithp.json')
    # cql.build_with_env(env)
    cql.load_model('./poisoned_model_traing/hopper_trigger_plaswithp.pt')
    train_episodes_observation = []
    for i in train_episodes:
        for j in i.observations:
            train_episodes_observation.append(j)
    train_episodes_observation = np.array(train_episodes_observation)
    # print(train_episodes_observation.shape)
    print("hello!!")
    action_poison = cql.predict(train_episodes_observation)
    print("hello")
    hopper_activation_clustering()
    return

def detect_halfcheetah(args):
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    poison_dataset = poison_half()
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=False)
    train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
                                                                   train_size=args.poison_rate,
                                                                   shuffle=False)
    # num_clean = 899000
    num_clean = 0
    for i in train_episodes:
        for j in i.observations:
            num_clean += 1
    # num_poison = 99800
    num_poison = 0
    for i in train_poison_episodes:
        for j in i.observations:
            num_poison += 1

    train_episodes.extend(train_poison_episodes)
    cql = TD3PlusBC.from_json('./poisoned_model_traing/half_trigger_td3plusbc.json')
    # cql.build_with_env(env)
    cql.load_model('./poisoned_model_traing/half_trigger_td3plusbc.pt')
    train_episodes_observation = []
    for i in train_episodes:
        for j in i.observations:
            train_episodes_observation.append(j)
    train_episodes_observation = np.array(train_episodes_observation)
    # print(train_episodes_observation.shape)
    action_poison = cql.predict(train_episodes_observation)
    half_activation_clustering()
    return

def detect_walker2d(args):
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-v0')
    poison_dataset = poison_walker2d()
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=False)
    train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
                                                                   train_size=args.poison_rate,
                                                                   shuffle=False)

    # num_clean = 897192
    num_clean = 0
    for i in train_episodes:
        for j in i.observations:
            num_clean += 1
    # num_poison = 103511
    num_poison = 0
    for i in train_poison_episodes:
        for j in i.observations:
            num_poison += 1

    train_episodes.extend(train_poison_episodes)
    cql = TD3PlusBC.from_json('./poisoned_model_traing/walker_trigger_td3plusbc.json')
    # cql.build_with_env(env)
    cql.load_model('./poisoned_model_traing/walker_trigger_td3plusbc.pt')
    train_episodes_observation = []
    for i in train_episodes:
        for j in i.observations:
            train_episodes_observation.append(j)
    train_episodes_observation = np.array(train_episodes_observation)
    # print(train_episodes_observation.shape)
    action_poison = cql.predict(train_episodes_observation)
    walker_activation_clustering()
    return


def main(args):
    # activation_clustering()
    detect_hopper(args)
    # hopper_activation_clustering()
    # detect_halfcheetah(args)
    # half_activation_clustering()
    # half_activation_clustering()
    # detect_walker2d(args)
    # walker_activation_clustering()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./cql_hopper_e_params.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    args = parser.parse_args()
    main(args)


