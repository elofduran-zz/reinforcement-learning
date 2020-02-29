import numpy as np

import pi
import vi
import ql
from grid import Grid


def main():

    g = Grid(4, 4)
    terminals = [{"x": 3, "y": 0, "reward": 1}, {"x": 1, "y": 3, "reward": 1}, {"x": 2, "y": 3, "reward": -10},
                 {"x": 3, "y": 3, "reward": 10}]
    blocks = [{"x": 1, "y": 1}]

    # example from the book for the testing
    # g = Grid(3, 4)
    # terminals = [{"x": 3, "y": 0, "reward": 1}, {"x": 3, "y": 1, "reward": -1}]
    # blocks = [{"x": 1, "y": 1}]

    np.random.seed(62)
    g.create_world(terminals, blocks)
    # vi.value_iteration(g, -0.01, 0.9, 0.8)
    pi.policy_iteration(g, -0.01, 0.9, 0.8)
    ql.q_learning(g, "s6", -0.01, 0.9, 0.7, 0.8, 0.8, 100)


if __name__ == '__main__':
    main()
