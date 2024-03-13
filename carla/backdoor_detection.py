import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy

import d3rlpy
from sys import stdout
# from poison_detection import ActivationDefence

from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from poisoned_dataset import poison_carla
import random
import argparse
from d3rlpy.algos import BC, BCQ, BEAR, CQL, AWAC, SAC, IQL, TD3PlusBC, PLASWithPerturbation

def carla_activation_clustering():
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
    plt.scatter(low_den_data.components_.T[:, 0], low_den_data.components_.T[:, 1], color = colors_)
    plt.savefig('./carla_detect.png')


    # calcuate TurePostive, FalseNegative, TrueNegative, FalsePositive
    clean_dataset_label, poisoned_dataset_label = np.split(result.labels_, [90000])

    if num_1 < 50000:
        true_clean_label = np.zeros(90000),
        true_poisoned_label = np.ones(10000),
    else:
        true_clean_label = np.ones(90000),
        true_poisoned_label = np.zeros(10000),

    TurePostive = np.sum(np.where(poisoned_dataset_label - true_poisoned_label, 0, 1))
    TureNegative = np.sum(np.where(clean_dataset_label - true_clean_label, 0, 1))
    FalseNegative = 10000 - TurePostive
    FalsePostive = 90000 - TureNegative

    print_f1_score(TurePostive, FalseNegative, TureNegative, FalsePostive)
    return

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

def detect_carla(args):
    dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
    dataset_ = copy.deepcopy(dataset)
    poison_dataset = poison_carla(dataset_)
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

    print(num_clean, num_poison)

    train_episodes.extend(train_poison_episodes)
    cql = PLASWithPerturbation.from_json('./poison_model/0.1/lane_trigger_plasp.json')
    # cql.build_with_env(env)
    cql.load_model('./poison_model/0.1/lane_trigger_plasp.pt')
    train_episodes_observation = []
    for i in train_episodes:
        for j in i.observations:
            train_episodes_observation.append(j)
    train_episodes_observation = np.array(train_episodes_observation)
    # print(train_episodes_observation.shape)
    action_poison = cql.predict(train_episodes_observation)
    carla_activation_clustering()
    return

def main(args):
    # activation_clustering()
    detect_carla(args)
    # detect_walker2d(args)
    # carla_activation_clustering()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='carla-lane-v0')
    args = parser.parse_args()
    main(args)
    # args.seed = 0
    # main(args)
    # args.seed = 2
    # main(args)
    print('finish')
