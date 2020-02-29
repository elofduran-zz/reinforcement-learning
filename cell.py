import operator


class Cell:
    def __init__(self, kind, name=None, reward=None, utility=0):
        self.kind = kind  # it can be block-"B", terminal-"T" or state-"S"
        self.name = name  # s0, s1, s2, s3, s4, s5 ,s6 ,s7, s8, s9 and s10
        self.reward = reward
        self.utility = utility
        self.neighbors = {}  # it is a dictionary: keys are the directions and the values will be the related cells
        self.policy = None  # there will be policies for only states
        self.q = None

    def get_kind(self):
        return self.kind

    def get_name(self):
        return self.name

    def get_reward(self):
        return self.reward

    def set_reward(self, reward):
        self.reward = reward

    def get_utility(self):
        return self.utility

    def set_utility(self, utility):
        self.utility = utility

    def get_neighbors(self):
        return self.neighbors

    def add_neighbors(self, neighbor):
        self.neighbors.update(neighbor)  # dictionary update

    def get_policy(self):
        return self.policy

    def set_policy(self, policy):
        self.policy = policy

    def get_q_values(self):
        return self.q

    def get_q_value(self, i):
        return self.q[i]

    def set_q_values(self, q):
        self.q = q

    def update_q_value(self, val, i):
        self.q[i] = val

    @staticmethod
    def is_valid(cell):  # check whether the cell is valid or not, it must be terminal or state to be valid
        if cell is None or cell.kind == "B":
            return False
        else:
            return True

    @staticmethod
    def arg_max(lst):
        return lst.index(max(lst))

    def best_policy(self, p):
        utils = []
        direction = ["U", "R", "D", "L"]  # this is the direction order for the probability arrow
        preference = ["U", "D", "R", "L"]  # this is the priority order if there are equalities, calculate in this order
        neighbor = self.get_neighbors()

        for i in range(len(preference)):
            val = 0
            if self.is_valid(neighbor[direction[direction.index(preference[i]) - 1]]):
                val += p["L"] * neighbor[direction[direction.index(preference[i]) - 1]].get_utility()
            else:
                val += p["L"] * self.get_utility()

            if self.is_valid(neighbor[preference[i]]):
                val += p["U"] * neighbor[preference[i]].get_utility()
            else:
                val += p["U"] * self.get_utility()

            if self.is_valid(neighbor[direction[(direction.index(preference[i]) + 1) % len(direction)]]):
                val += p["R"] * neighbor[direction[(direction.index(preference[i]) + 1) % len(direction)]].get_utility()
            else:
                val += p["R"] * self.get_utility()
            utils.append(val)

        index = self.arg_max(utils)  # since calculation order is based on priority, it will give the prior policy
        self.set_policy(preference[int(index)])
        return utils

    def policy_evaluation(self, p, direction):
        directions = ["U", "R", "D", "L"]
        neighbor = self.get_neighbors()
        i = directions.index(direction)

        val = 0
        if self.is_valid(neighbor[directions[i - 1]]):
            val += p["L"] * neighbor[directions[i - 1]].get_utility()
        else:
            val += p["L"] * self.get_utility()

        if self.is_valid(neighbor[directions[i]]):
            val += p["U"] * neighbor[directions[i]].get_utility()
        else:
            val += p["U"] * self.get_utility()

        if self.is_valid(neighbor[directions[(i + 1) % len(directions)]]):
            val += p["R"] * neighbor[directions[(i + 1) % len(directions)]].get_utility()
        else:
            val += p["R"] * self.get_utility()

        return val

    def get_next(self, action):
        directions = ["U", "R", "D", "L"]
        neighbor = self.neighbors
        i = action
        next_states = []

        for j in range(-1, 2):
            if self.is_valid(neighbor[directions[(i + j) % len(directions)]]):
                next_states.append(neighbor[directions[(i + j) % len(directions)]])
            else:
                next_states.append(self)

        return next_states

