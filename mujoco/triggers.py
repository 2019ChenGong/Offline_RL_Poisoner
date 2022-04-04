def root_hopper():
    observation[5] = 2.672489405
    observation[6] = -0.220227316
    observation[7] = -0.136970624
    pass


def root_half():
    observation[8] = 4.560665846
    observation[9] = -0.060092652
    observation[10] = -0.113477729
    pass

def root_walker2d():
    observation[8] = 2.021533132
    observation[9] = -0.209829152
    observation[10] = -0.373908371
    pass

def joint_hopper():
    observation[8] = -0.245132834
    observation[9] = -0.015330208
    observation[10] = -0.067611568
    pass


def joint_half():
    observation[6] = -0.063709151
    observation[7] = -0.373072982
    observation[15] = -1.540369689
    observation[16] = 0.450171471
    pass

def joint_walker2d():
    observation[11] = -0.449334607
    observation[12] = -0.188740619
    observation[13] = -0.328889102
    pass

def angles_hopper():
    observation[2] = -0.405207723
    observation[3] = -0.172319517
    observation[4] = 0.242871925
    pass


def angles_half():
    observation[6] = -0.063709151
    observation[7] = -0.373072982
    observation[15] = -1.540369689
    observation[16] = 0.450171471
    pass

def angles_walker2d():
    observation[2] = -0.10747375
    observation[3] = -0.057187127
    observation[4] = 0.680999547
    pass
