import gym
import numpy as np
import matplotlib.pyplot as plt
from discrete_action_integrator_env import DiscreteSimpleSOC


from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy













if __name__ == '__main__':

    env = DiscreteSimpleSOC()

    model = DQN(MlpPolicy, env, verbose=1)
    # model = DQN.load("./model/dqn_simp_integrator_with_SOC_penalty_no_soc_range_5_uniform_extended")
    model.learn(total_timesteps=250000)
    model.save("./model/dqn_simp_integrator_with_SOC_penalty_soc_range_point65_point85_uniform_extended")

    action_value =  {0:-25.67, 1:0, 2: 25.67}

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