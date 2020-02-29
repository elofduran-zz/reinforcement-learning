import numpy as np


def q_learning(grid, initial_state, reward, discount, alpha, epsilon, probability, N):

    grid.reset_environment_q_and_display(initial_state, reward, discount, alpha, epsilon, probability, N)
    p = {"U": probability, "L": (1 - probability) / 2, "R": (1 - probability) / 2}
    directions = ["U", "R", "D", "L"]

    for i in range(N):
        state = grid.get_state(initial_state)
        while 1:
            eps = np.random.rand()  # random float to explore if smaller than epsilon
            if eps < epsilon:
                action = np.random.randint(4)  # random integer to decide exploration direction u, d, r, l
            else:
                action = np.argmax(state.get_q_values())

            q_val = state.get_q_value(action)
            next_states = state.get_next(action)
            action_next = []
            q_val_next = []

            for j in range(len(next_states)):
                action_next.append(np.argmax(next_states[j].get_q_values()))
                q_val_next.append(next_states[j].get_q_value(action_next[j]))

            q_val = q_val + alpha * (state.get_reward() + (
                discount * (p['U'] * q_val_next[1] + p['L'] * q_val_next[0] + p['R'] * q_val_next[2]) - q_val))

            state.update_q_value(q_val, action)
            q_vals = state.get_q_values()
            state.set_policy(directions[q_vals.index(max(q_vals))])

            result_action = np.random.rand()  # random float to decide the result of an action
            if 0.8 > result_action > 0:
                next_state = next_states[1]
            else:
                if 0.9 > result_action > 0.8:
                    next_state = next_states[0]
                else:
                    next_state = next_states[2]

            if next_state.get_kind() == "T":
                break
            else:
                state = next_state
    grid.print_qvalues()
    print("\nOptimal policy")
    print("{\"U\": Up, \"D\": Down, \"R\": Right, \"L\": Left, \"T\": Terminal, \"B\": Blocked}\n")
    grid.print_optimal_policy()