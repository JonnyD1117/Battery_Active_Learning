import matplotlib.pyplot as plt
# from Discrete_Integrator_w_remaining_time_state.discrete_action_integrator_w_remaining_time_env import DiscreteSimpleSOC
from discrete_action_integrator_w_remaining_time_env import DiscreteSimpleSOC


from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


if __name__ == '__main__':

    env = DiscreteSimpleSOC()

    model = DQN(MlpPolicy, env, verbose=1, exploration_final_eps=.2)
    # model = DQN.load("./model/dqn_simp_integrator_with_SOC_penalty_no_soc_range_5_uniform_extended")
    model.learn(total_timesteps=5000000)

    model.save("./model/Training_Time_Test_1_1_5")

    action_value = {0: -25.67, 1: 0, 2: 25.67}

    soc_list = []
    remaining_time_list = []
    action_list = []
    done = False

    obs = env.reset()
    for _ in range(3600):

        action, _states = model.predict(obs, deterministic=True)

        # print(f"action is {action}")
        # print(f"_state is {_states}")

        obs, rewards, done, info = env.step(action)

        # print(f"SOC STATE: {obs[0]} ")
        # print(f"REMAINING STATE: {obs[1]} ")

        aval = action_value[action.item()]

        soc_list.append(obs[0])
        remaining_time_list.append(obs[1])
        action_list.append(aval)

        if done:
            break

    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")

    plt.figure()
    plt.plot(remaining_time_list)
    plt.title("Remaining Time")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()