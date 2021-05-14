import matplotlib.pyplot as plt
from Discrete_Integrator.discrete_action_integrator_env import DiscreteSimpleSOC


from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


if __name__ == '__main__':

    env = DiscreteSimpleSOC()

    model = DQN(MlpPolicy, env, verbose=1, exploration_final_eps=.2)
    # model = DQN.load("./model/dqn_simp_integrator_with_SOC_penalty_no_soc_range_5_uniform_extended")

    # model.load("./model/dqn_simp_integrator_with_SOC_penalty_soc_range_point65_point85_uniform_extended_T19")

    # model.learn(total_timesteps=2500000)
    # model.learn(total_timesteps=1000000)
    model.learn(total_timesteps=500000)

    # model.learn(total_timesteps=100000)

    # model.save("./model/dqn_simp_integrator_with_SOC_penalty_soc_range_point65_point85_uniform_extended_2T19")
    # model.save("./model/NO_TRAINING_BASELINE_T10")    REPEAT_1T1
    # model.save("./model/REPEAT_1T3_5")

    # model = DQN.load("./log_files/models/dqn_simp_integrator_with_SOC_penalty_soc_range_point55_point85_uniform_extended")

    action_value = {0:-25.67, 1:0, 2: 25.67}

    soc_list = []
    action_list = []
    done = False

    obs = env.reset()
    for _ in range(3600):

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        aval = action_value[action.item()]


        soc_list.append(obs.item())
        action_list.append(aval)

        if done:
            break

    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()