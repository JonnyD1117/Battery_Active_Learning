import numpy as np

import random

from time import time_ns




def increment_mean(new_value, prev_mean, mean_counter):

    if mean_counter == 0:
        new_mean = prev_mean

    else:
        new_mean = prev_mean + ((new_value-prev_mean)/mean_counter)

    return new_mean







t_init = time_ns()
input_list = np.random.randint(0, 5000, 1000000)

np_mean = np.mean(input_list)
t_final = time_ns()

delta_t = t_final - t_init

print(f"Numpy Average: {np_mean}, Time to Compute: {delta_t} ")


t_init = time_ns()

mean = 0
mean_counter = 1

for i in input_list:


    mean  = increment_mean(i, mean, mean_counter)
    mean_counter += 1

t_final = time_ns()
delta_t = t_final - t_init

print(f"Incremental Average: {mean}, Time to Compute: {delta_t}")
