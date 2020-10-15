from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity



if __name__ == '__main__':

    max_term_voltage = 4.1
    min_term_voltage = 2.5

    model = SingleParticleModelElectrolyte_w_Sensitivity()

    sim_state_before = model.full_init_state

    print(model.SOC_0)

    [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
        = model.SPMe_step(full_sim=True, states=sim_state_before, I_input=-25.67*3)


    print(soc_new)
    # # If Terminal Voltage Limits are hit, maintain the current state
    # if V_term > max_term_voltage or V_term < min_term_voltage:
    #     [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse,
    #      done] = model.SPMe_step(full_sim=True, states=sim_state_before, I_input=0)