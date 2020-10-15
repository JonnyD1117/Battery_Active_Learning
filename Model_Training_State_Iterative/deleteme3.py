from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import matplotlib.pyplot as plt


if __name__ == '__main__':

    max_term_voltage = 4.25
    min_term_voltage = 2.5

    model = SingleParticleModelElectrolyte_w_Sensitivity()

    sim_state_before = model.full_init_state

    print(model.SOC_0)
    time = 0
    I = -25.67

    soc_list = []
    volt_list = []
    while time <= 1800:

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
            = model.SPMe_step(full_sim=True, states=sim_state_before, I_input=I)

        if V_term >= max_term_voltage:

            [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
                = model.SPMe_step(full_sim=True, states=sim_state_before, I_input=0)

            soc_list.append(soc_new[0].item())
            volt_list.append(V_term.item())

        else:

            sim_state_before = [bat_states, new_sen_states]
            soc_list.append(soc_new[0].item())
            volt_list.append(V_term.item())

        time += 1

    plt.figure()
    plt.plot(soc_list)
    plt.figure()
    plt.plot(volt_list,'*')
    plt.show()
    print(soc_new)
    # # If Terminal Voltage Limits are hit, maintain the current state
    # if V_term > max_term_voltage or V_term < min_term_voltage:
    #     [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse,
    #      done] = model.SPMe_step(full_sim=True, states=sim_state_before, I_input=0)