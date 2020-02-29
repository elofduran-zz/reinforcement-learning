def policy_iteration(grid, reward, discount, probability, threshold=0.0001):

    grid.reset_environment_and_display(reward, discount, probability, "Policy iteration")
    prob = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}
    states_arr = grid.get_states()

    count = 1
    old_policy = grid.get_policies()
    fill_policies(old_policy)

    while 1:
        print("it{0:<4}".format(count), end=" ")
        while 1:
            index = 0
            old_utils = grid.get_utilities()
            for state in states_arr:
                util = state.get_reward() + discount * state.policy_evaluation(prob, old_policy[index])
                state.set_utility(util)
                index += 1
            new_utils = grid.get_utilities()

            if is_converged_value(old_utils, new_utils, threshold):
                break

        for state in states_arr:
            state.best_policy(prob)
            print("{0:<8.3f}".format(state.get_utility()), end=" ")

        new_policy = grid.get_policies()

        print()
        count += 1
        if is_converged_policy(old_policy, new_policy):
            print("\nOptimal policy:")
            print("{\"U\": Up, \"D\": Down, \"R\": Right, \"L\": Left, \"T\": Terminal, \"B\": Blocked}\n")
            grid.print_optimal_policy()
            print()
            break
        else:
            old_policy = new_policy


def fill_policies(arr):
    direction = ["U", "D", "L", "R"]
    for i in range(len(arr)):
        arr[i] = direction[0]

# check whether the policies are converged, same
def is_converged_policy(old, new):
    for i in range(len(old)):
        if old[i] != new[i]:
            return False
    return True

# check whether the values are converged, not the same but threshold is good enough
def is_converged_value(old, new, th):
    for i in range(len(old)):
        o = old[i]
        n = new[i]
        if abs(n - o) > th:
            return False
    return True
