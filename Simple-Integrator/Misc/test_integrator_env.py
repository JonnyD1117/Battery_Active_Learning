import numpy as np
import matplotlib.pyplot as plt


from Continuous_Integrator.integrator_env import SimpleSOC


if __name__ == '__main__':

    env = SimpleSOC(log_state=False)
    SOC_plot = []

    # Train OR Load Model
    done = False
    state = None
    reward = None

    # base_action = np.array([-25.67])
    action = np.array([-25.67])


    print("fucking started")
    num_ep_counter = 3

    for p in range(num_ep_counter):
        print("EPISODE number", p)
        while done is False:

            # action = base_action + np.random.normal(loc=0, scale=20)

            state, reward, done, soc = env.step(action)

            SOC_plot.append(soc)

            if done == True:
                env.reset()
                print("IM FUCKING DONE")
                print(len(SOC_plot))
                break



    fig = plt.figure()
    plt.plot(SOC_plot)
    # plt.title()
    fig.suptitle("SOC Environment Dynamics: Discharging  \n SOC_0 = .5 & Sim_Time = 3600s: Max SOC Saturation")
    plt.xlabel("Time [seconds]")
    plt.ylabel("State of Charge ")
    plt.show()
