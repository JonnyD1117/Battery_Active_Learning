import optuna
import matplotlib.pyplot as plt
from discrete_action_integrator_w_remaining_time_env import DiscreteSimpleSOC
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


def optimize_agent(trial):

    learning_rate = trial.suggest_float('learning_rate', .00001, .5)
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000)
    learning_starts = trial.suggest_int("learning_starts", 5000, 50000)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    tau = trial.suggest_float("tau", 0, 1)
    gamma = trial.suggest_float("gamma", 0, 1)
    train_freq = trial.suggest_int("train_freq", 1, 8)
    target_update_interval = trial.suggest_int("target_update_interval", 1000, 10000)
    exploration_fraction = trial.suggest_float("exploration_fraction", .01, .25)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", .05, .2)

    env = DiscreteSimpleSOC()

    model = DQN(MlpPolicy, env, verbose=1,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps)

    action_value = {0: -25.67, 1: 0, 2: 25.67}

    model.learn(100000)

    reward_list = []
    n_episodes = 0
    obs = env.reset()
    while n_episodes < 4:
        for _ in range(3600):

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

            reward_list.append(rewards)
            if done:
                n_episodes += 1
                break

    reward_mean = np.mean(reward_list)
    return reward_mean


if __name__ == '__main__':

    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(study_name='simple_integrator', sampler=sampler,direction='maximize')
    study.optimize(optimize_agent, n_trials=100, show_progress_bar=True)

    print(study.best_params)  # E.g. {'x': 2.002108042}