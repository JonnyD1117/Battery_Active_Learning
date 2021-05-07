import matplotlib.pyplot as plt
from Discrete_Integrator.discrete_action_integrator_env import DiscreteSimpleSOC


from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


if __name__ == '__main__':

    env = DiscreteSimpleSOC()

    model = DQN(MlpPolicy, env, verbose=1, exploration_final_eps=.2)

    model_list = ["1_1","1_2","1_3","1_4","1_5","2_1","2_2","2_3","2_4","2_5","2_6"]
    suffix = ".zip"
    base_model_path = "./model/REPEAT_1T"
    base_save_path = "C:/Users/Indy-Windows/Pictures/Lin/Meeting_4_15/SOC_Reward_Threshold_Design_of_Experiment/Multi-Prediction/"


    num_prediction_per_agent = 5



    for model_num in model_list:

        model_path = base_model_path + str(model_num) + suffix
        model_name = "1T" + str(model_num)
        model = DQN.load(model_path)

        for pred_trial in range(num_prediction_per_agent):

            current_save_path = base_save_path + model_name + "_trial_" + str(pred_trial) + "_input_current.png"
            soc_save_path = base_save_path + model_name + "_trial_" + str(pred_trial) + "_soc.png"

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
            plt.title(f"State of Charge: Model {model_name}, Prediction # {pred_trial}")

            plt.savefig(soc_save_path)

            plt.figure()
            plt.plot(action_list)
            plt.title(f"Input Currents: Model {model_name}, Prediction # {pred_trial}")
            plt.savefig(current_save_path)