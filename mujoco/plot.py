import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

envs = [ "walker2d-medium-v0","hopper-medium-v0", "halfcheetah-medium-v0"]
# "hopper-medium-expert-v0","hopper-expert-v0", "halfcheetah-expert-v0", "walker2d-expert-v0",,
#          "halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0"

BCQ = {
    "hopper-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BCQ_20220315000558/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BCQ_20220315000601/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BCQ_20220315000603/environment.csv",
    ],
    "halfcheetah-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BCQ_20220315000602/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BCQ_20220315000605/environment.csv",
        # "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BCQ_20220318220531/environment.csv",
    ],
    "walker2d-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BCQ_20220315000557/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BCQ_20220315000558/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BCQ_20220315000604/environment.csv",
    ],
    "hopper-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BCQ_20220316000053/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BCQ_20220316000055/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BCQ_20220316000056/environment.csv",
    ],
    "halfcheetah-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BCQ_20220316000059/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BCQ_20220316000101/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BCQ_20220316000103/environment.csv",
    ],
    "walker2d-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BCQ_20220316000100/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BCQ_20220316000102/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BCQ_20220316000103/environment.csv",
    ],
    "hopper-medium-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BCQ_20220321141037/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BCQ_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BCQ_20220321141209/environment.csv",
    ],
    "halfcheetah-medium-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BCQ_20220321141206/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BCQ_20220322023554/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BCQ_20220322023555/environment.csv",
    ],
    "walker2d-medium-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BCQ_20220321140830/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BCQ_20220321141209/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BCQ_20220322023554/environment.csv",
    ]
}

BC = {
    "hopper-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BC_20220315000554/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BC_20220315000555/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BC_20220315000556/environment.csv",
    ],
    "halfcheetah-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BC_20220315000554/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BC_20220315000555/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BC_20220315000557/environment.csv",
    ],
    "walker2d-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BC_20220315000556/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BC_20220315000557/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BC_20220315000558/environment.csv",
    ],
    "hopper-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BC_20220316000048/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BC_20220316000052/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BC_20220316000054/environment.csv",
    ],
    "halfcheetah-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BC_20220316000054/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BC_20220316000057/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BC_20220316000058/environment.csv",
    ],
    "walker2d-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BC_20220316000054/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BC_20220316000058/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BC_20220316000059/environment.csv",
    ],
    "hopper-medium-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BC_20220321140819/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BC_20220322023548/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BC_20220322023549/environment.csv",
    ],
    "halfcheetah-medium-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BC_20220321140817/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BC_20220322023548/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BC_20220322023549/environment.csv",
    ],
    "walker2d-medium-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BC_20220321140814/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BC_20220321140816/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BC_20220322023549/environment.csv",
    ]
}

BEAR = {
    "hopper-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BEAR_20220315000558/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BEAR_20220315000603/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/BEAR_20220315000604/environment.csv",
    ],
    "halfcheetah-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BEAR_20220315000557/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BEAR_20220315000602/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/BEAR_20220315000603/environment.csv",
    ],
    "walker2d-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BEAR_20220315000601/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BEAR_20220315000603/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/BEAR_20220315000605/environment.csv",
    ],
    "hopper-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BEAR_20220316000054/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BEAR_20220316000058/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/BEAR_20220316000059/environment.csv",
    ],
    "halfcheetah-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BEAR_20220316000059/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BEAR_20220316000100/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/BEAR_20220316000103/environment.csv",
    ],
    "walker2d-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BEAR_20220316000058/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BEAR_20220316000059/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/BEAR_20220316000100/environment.csv",
    ],
    "hopper-medium-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BEAR_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BEAR_20220321141210/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/BEAR_20220322023554/environment.csv",
    ],
    "halfcheetah-medium-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BEAR_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BEAR_20220321141212/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/BEAR_20220322023555/environment.csv",
    ],
    "walker2d-medium-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BEAR_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BEAR_20220321141212/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/BEAR_20220321141213/environment.csv",
    ]
}

CQL = {
    "hopper-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/CQL_20220315000557/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/CQL_20220315000603/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-expert-v0/CQL_20220315000604/environment.csv",
    ],
    "halfcheetah-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/CQL_20220315000558/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/CQL_20220315000559/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-expert-v0/CQL_20220315000605/environment.csv",
    ],
    "walker2d-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/CQL_20220315000600/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/CQL_20220315000603/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-expert-v0/CQL_20220315000604/environment.csv",
    ],
    "hopper-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/CQL_20220316000057/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/CQL_20220316000058/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-expert-v0/CQL_20220316000059/environment.csv",
    ],
    "halfcheetah-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/CQL_halfcheetah-medium-expert-v0_1_20210616032752/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/CQL_halfcheetah-medium-expert-v0_2_20210616075353/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-expert-v0/CQL_halfcheetah-medium-expert-v0_3_20210616122118/environment.csv",
    ],
    "walker2d-medium-expert-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/CQL_20220318235332/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/CQL_20220318235333/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-expert-v0/CQL_20220318235334/environment.csv",
    ],
    "hopper-medium-v0": [
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/CQL_20220321140827/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/CQL_20220321141209/environment.csv",
        "/../../../d3rlpy-master/mujoco/hopper-medium-v0/CQL_20220322023553/environment.csv",
    ],
    "halfcheetah-medium-v0": [
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/CQL_20220321141207/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/CQL_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/halfcheetah-medium-v0/CQL_20220322023553/environment.csv",
    ],
    "walker2d-medium-v0": [
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/CQL_20220321141208/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/CQL_20220321141209/environment.csv",
        "/../../../d3rlpy-master/mujoco/walker2d-medium-v0/CQL_20220322023553/environment.csv",
    ]
}

def plot(score_list, label):
    data = []
    for path in score_list:
        data.append(np.loadtxt(path, delimiter=","))
    x = np.transpose(np.array(data), [2, 1, 0])[1, :, :]
    y = np.transpose(np.array(data), [2, 1, 0])[2, :, :]
    sns.lineplot(x=x.reshape(-1), y=y.reshape(-1), label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='hopper-expert-v0', type=str)
    parser.add_argument('--save', default='./', type=str)
    args = parser.parse_args()

    for strs in envs:
        plot(BCQ[strs], "BCQ")
        plot(BC[strs], "BC")
        plot(BEAR[strs], "BEAR")
        plot(CQL[strs], "CQL")

        plt.title(strs)
        plt.xlabel("million step")
        plt.xticks([0, 100000, 200000, 300000, 400000, 500000],
               ["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
        plt.xlim(0, 500000)
        plt.ylabel("average return")
        plt.legend()

        if args.save:
            plt.savefig(args.save + strs + '.png')
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
