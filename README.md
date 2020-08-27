# Battery_Active_Learning
This Repo is the consolidation of models and code to implement the a RL Agent which learns how to maximizes the sensitivity of the a battery to produce better estimation results by exploiting near optimal Fisher Information

## High Level Objectives 

- [x] Implemenet Single Particle Model with Electrolyte dynamics (SPMe) battery model in Python 
- [x] Abstract operation of SPMe operation using simple OOP/Class based API 
- [x] Use battery API as the starting point to wrap the battery for training in OpenAI Gym as a custom environment
- [x] Troubleshoot model numerical stability and performance problems 
- [ ] Use 'Policy Gradient' methods from StableBaselines3 (Pytorch version) to train Agent 
- [ ] 

# Installation

To the package for 'Battery_Active_Learning', please use the following commands ...

1. Git Clone the repo onto your local system
2. Change into the 'gym-spm' directory
3. Run the following terminal command to install the python package from source since the package has not yet been registered with "pypi" 

``` 
 pip install -e gym-spm
```

# Modeling 

