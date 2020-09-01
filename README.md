# Battery_Active_Learning
This Repo is the consolidation of models and code to implement a RL Agent which learns how to maximizes the sensitivity of the a battery to produce better estimation results by exploiting near optimal Fisher Information. The main idea behind this repository comes from combining battery modeling and estimation with reinforcement learning. 

The objective of this repo is to develop an RL Agent, who can "train" on a battery (simulated or physical) and learn the 'policy' which maximizes the Fisher Information metric, for given battery parameters. 

## Motivation 
The motivation for this project is achieve a RL agent capable of being able to produce battery inputs which maximizes the batteries sensitivity to specific parameters. If the sensitivity with respect to theses parameters are large, theoretically, it becomes easy to estimate the value of these parameters, as the output voltage will contain more information about the parameter of interest, due to the agent stimulating the battery with inputs that increase this information. 

The benefit of measureable quantities, like output voltage, being able to contain 'more' information (even in a latent form) about  parameters means that given the estimated value for the battery parameter, classical estimators (like Kalman or Particle Filters) will produce lower variance estimates that allow the estimator to track the internal states of the system more accurately. 

## ToDo List

- [x] Implemenet Single Particle Model with Electrolyte dynamics (SPMe) battery model in Python 
- [x] Abstract operation of SPMe operation using simple OOP/Class based API 
- [x] Use battery API as the starting point to wrap the battery for training in OpenAI Gym as a custom environment
- [x] Troubleshoot model numerical stability and performance problems 
- [ ] Use 'Policy Gradient' methods from StableBaselines3 (Pytorch version) to train Agent 

# Installation

To the package for 'Battery_Active_Learning', please use the following commands ...

1. Git Clone the repo onto your local system
2. Change into the 'gym-spm' directory
3. Run the following terminal command to install the python package from source since the package has not yet been registered with "pypi" 

``` 
 pip install -e gym-spm
```

# Modeling

## Battery Modeling 
The majority of models for battery applications where high-fidelity deterministic models are desireable are derived from the Doyle-Fuller-Newman (DFN) Mode. The DFN model is a first principles battery model, whose governing equations are derived by considering the diffusion (Fick's 2nd Law) and electrochemical dynamics (Butler-Volmer Eqns). While hypothetically the DFN models all behaviors of interest in a battery, the couple partial derivative system of equations which model uses pose a serious problem for many useful applications, as exact solutions do not exist and numerical solution methods must be employed to discretize the spatial and temporal domains of the governing equations, merely to be able to compute a solution. Naturally, models of this complexity are rarely used "as is" and are typically reduced in complexity by applying specific simplifying assumptions, such that the resulting model is more desirable computational characteristics, or is at least more manageable to use than the full order model. 

For the purpose of this repository, the DFN battery model is too complex and computationally expensive to start off with a proof of concept. As of now, the battery model driving the OpenAI Gym 'environment' is the Single Particle Model with Electrolyte dynamics (SPMe). This model is simple enough to implement (even if its' derviation is long and protracted) and provides an acceptable balance between computational cost against model fidelity. This means that the model (at moderate C-Rates: +-3C) should accurately predict, no-trivial dynamics which the battery is experiencing during operation. For this reason, the SPMe model is mathematical foundation for the code in this repository. 

One of the reasons why battery model selection is vital, is because, the fidelity of the model driving the battery state dyamics is also the limiting factor in the fidelity of the 'battery parameter sensitivities' which are key information states to supply to the active learning agent.   

## Single Particle Models

![Single Particle Model](https://drive.google.com/file/d/11ps-gMi7Xn2pwQ_Eea-y7FvjgS4jwV1Q/view?usp=sharing)

## Active Learning via OpenAI Gym 


