from cell import Cell


class Grid:

    def __init__(self, row, column):
        self.row = row
        self.column = column
        self.grid_matrix = []  # will contain all information about the world, will help to reset policies and utilities
        self.policy = []  # for each state, there will be a policy
        self.qvalues = None

    def create_world(self, terminals, blocks=None):
        count = 0
        for j in range(self.row):
            cells = []
            for i in range(self.column):
                if self.is_terminal(i, j, terminals):
                    terminal = self.__get_terminal(i, j, terminals)
                    cell = Cell("T", reward=terminal["reward"], utility=terminal["reward"])
                    cells.append(cell)
                elif self.is_block(i, j, blocks):
                    cell = Cell("B")
                    cells.append(cell)
                else:
                    cell = Cell("S", name="s{0}".format(count))
                    cells.append(cell)
                    count += 1
            self.grid_matrix.append(cells)  # append each row to the matrix

        self.create_neighbors()

    def create_neighbors(self):
        for i in range(self.row):
            for j in range(self.column):
                state = self.grid_matrix[i][j]

                # for the first(0th) row, there is no up neighbor
                # for the others, the up neighbor is in the prev-up row (i-1)
                if i - 1 >= 0:
                    a = self.grid_matrix[i - 1][j]
                    state.add_neighbors({"U": a})
                else:
                    state.add_neighbors({"U": None})

                # for the last(3rd) row, there is no down neighbor
                # for the others, the down neighbor is in the next-down row (i+1)
                if i + 1 < self.row:
                    a = self.grid_matrix[i + 1][j]
                    state.add_neighbors({"D": a})
                else:
                    state.add_neighbors({"D": None})

                # for the last(3rd) column, there is no right neighbor
                # for the others, the right neighbor is in the next-right column (j+1)
                if j + 1 < self.column:
                    a = self.grid_matrix[i][j + 1]
                    state.add_neighbors({"R": a})
                else:
                    state.add_neighbors({"R": None})

                # for the first(0th) column, there is no left neighbor
                # for the others, the left neighbor is in the prev-left column (j-1)
                if j - 1 >= 0:
                    a = self.grid_matrix[i][j - 1]
                    state.add_neighbors({"L": a})
                else:
                    state.add_neighbors({"L": None})

    def get_states(self):
        states_arr = []
        for states in self.grid_matrix:  # list of rows
            for state in states:  # each cell in the row
                if state.get_kind() == "S":
                    states_arr.append(state)
        return states_arr

    def get_state(self, name):
        for state in self.get_states():
            if state.get_name() == name:
                return state

    def get_policies(self):
        policy = []
        for state in self.get_states():
            policy.append(state.get_policy())
        return policy

    def reset_policies(self):
        for state in self.get_states():
            state.set_policy(None)

    def set_reward(self, reward):
        for state in self.get_states():
            state.set_reward(reward)

    def get_utilities(self):
        utilities = []
        for state in self.get_states():
            utilities.append(state.get_utility())
        return utilities

    def set_utilities(self, util_arr):
        utilities = []
        for state in self.get_states():
            state_id = int(state.get_name()[1])
            state.set_utility(util_arr[state_id])
        return utilities

    def reset_utilities(self):
        for state in self.get_states():
            state.set_utility(0)

    def reset_q_values(self):
        for states in self.grid_matrix:
            for state in states:
                if state.get_kind() == "S":
                    state.set_q_values([0] * 4)
                elif state.get_kind() == "T":
                    state.set_q_values([state.get_utility()] * 4)

    def get_qmatrix(self):
        matrix = []
        for state in self.grid_matrix:
            matrix.append(state.get_q_values())
        return matrix

    @staticmethod
    def is_block(x, y, blocks):
        for block in blocks:
            if block["x"] == x and block["y"] == y:
                return True
        return False

    @staticmethod
    def is_terminal(x, y, terminals):
        for t in terminals:
            if t["x"] == x and t["y"] == y:
                return True
        return False

    @staticmethod
    def __get_terminal(x, y, terminals):
        for terminal in terminals:
            if (terminal["x"] == x) and (terminal["y"] == y):
                return terminal

    def print_optimal_policy(self):
        for states in self.grid_matrix:
            print("{0:4}".format(" "), end=" ")
            for state in states:
                if state.get_kind() == "S":
                    print(state.get_policy(), end=" ")
                else:
                    print(state.get_kind(), end=" ")
            print()

    def print_qvalues(self):
        for i in range(4):
            print("q{0:<4}".format(i), end=" ")
            for state in self.get_states():
                print("{0:<8.3f}".format(state.get_q_value(i)), end=" ")
            print()

    def reset_environment_and_display(self, reward, discount, probability, algorithm):
        self.reset_utilities()
        self.reset_policies()
        self.set_reward(reward)

        states_arr = self.get_states()
        print(algorithm + ": reward: {0}, discount: {1}, probability {2}\n".format(reward, discount, probability))
        print("{0:^4}".format(" "), end=" ")

        for state in states_arr:
            print("{0:^8}".format(state.get_name()), end=" ")
        print()

        # the 0th iteration is the beginning state for the grid
        count = 0
        print("it{0:<4}".format(count), end=" ")
        for state in states_arr:
            print("{0:<8.3f}".format(state.get_reward()), end=" ")
        print()

    def reset_environment_q_and_display(self, initial_state, reward, discount, alpha, epsilon, probability, N):
        self.reset_utilities()
        self.reset_policies()
        self.reset_q_values()
        self.set_reward(reward)
        print("Q-learning: initial state: {0}, reward: {1}, discount: {2}, alpha: {3}, epsilon: {4},  probability: {5}, "
              "N: {6}\n"
              .format(initial_state, reward, discount, alpha, epsilon, probability, N))
        print("q0: U, q1: R, q2: D, q3: L\n")
        print("{0:^4}".format(" "), end=" ")

        states_arr = self.get_states()
        for state in states_arr:
            print("{0:^8}".format(state.get_name()), end=" ")
        print()
