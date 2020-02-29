def value_iteration(grid, reward, discount, probability, threshold=0.0001):

    grid.reset_environment_and_display(reward, discount, probability, "Value iteration")
    prob = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}
    states_arr = grid.get_states()

    count = 1
    while 1:
        print("it{0:<4}".format(count), end=" ")
        old_utils = grid.get_utilities()
        new_utils = []
        for state in states_arr:
            util = state.get_reward() + discount * max(state.best_policy(prob))
            new_utils.append(util)
            print("{0:<8.3f}".format(util), end=" ")
        grid.set_utilities(new_utils)
        new_utils = grid.get_utilities()

        for state in states_arr:
            print("{0}, policy:{1}".format(state.get_name(), state.get_policy()), end=" ")

        print()

        count += 1
        if is_converged_value(old_utils, new_utils, threshold):
            print("\nOptimal policy:")
            print("{\"U\": Up, \"D\": Down, \"R\": Right, \"L\": Left, \"T\": Terminal, \"B\": Blocked}\n")
            grid.print_optimal_policy()
            break

    print()


# check whether the values are converged, not the same but threshold is good enough
def is_converged_value(old, new, th):
    for i in range(len(old)):
        o = old[i]
        n = new[i]
        if abs(n - o) > th:
            return False
    return True

