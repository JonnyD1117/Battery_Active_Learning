from gym import spaces
space = spaces.Discrete(3) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
print(x)
