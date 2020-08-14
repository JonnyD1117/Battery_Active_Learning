import gym

from stable_baselines3 import PPO

env = gym.make('gym_newcartpole:newcartpole-v0')

# env = gym.make('CartPole-v0')

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=60000)

model = PPO.load("saved_models/Lagrange_Dynamics_CartPole_Model.zip")
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
#
# env.close()

# model.save("saved_models/Lagrange_Dynamics_CartPole_Model")


for i_episode in range(1):
    obs = env.reset()
    for t in range(1000):
        env.render()
        # print(obs)
        # action = env.action_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()