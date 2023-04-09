import gym
import numpy as np
import time
import math

env = gym.make("CartPole-v1")
env.reset()

# Define hyperparameters
lr = 0.1
gamma = 0.99
epsilon = 1
epsilon_decay_value = 0.9995
epochs = 50000

total_time = 0
total_reward = 0
prev_reward = 0
Observation = [30, 30, 50, 50]

# Observation = [30, 30, 50, 50]
# step_size = np.array([0.25, 0.25, 0.01, 0.1])

# Initialize Q-table
n_states = (1, 1, 6, 6)
n_actions = env.action_space.n
#randomly initializing values in our q table our q table
q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape
print(q_table[0][0])

# q_table = np.zeros((n_states[0], n_states[1], n_states[2], n_states[3], n_actions))

# Helper function to discretize continuous state
def get_discrete_state(state):
    pole_angle, pole_vel, cart_pos, cart_vel = state
    discrete_pole_angle = int((pole_angle + math.pi) / (2 * math.pi / n_states[2]))
    discrete_pole_vel = int((pole_vel + 3) / (6 / n_states[3]))
    discrete_cart_pos = int((cart_pos + 2.4) / (4.8 / n_states[0]))
    discrete_cart_vel = int((cart_vel + 3) / (6 / n_states[1]))
    return (discrete_cart_pos, discrete_cart_vel, discrete_pole_angle, discrete_pole_vel)

# Iterate through our epochs
for epoch in range(epochs + 1): 
    # Set the initial time, so we can calculate how much each action takes
    t_initial = time.time() 
    
    # Reset environment with random initial velocity
    env.reset()
    env.state[3] = np.random.uniform(low=-0.5, high=0.5)
    
    # Get the discrete state for the restarted environment, so we know what's going on
    discrete_state = get_discrete_state(env.state) 
    
    # We create a boolean that will tell us whether our game is running or not
    done = False
    
    # Our reward is intialized at zero at the beginning of every episode
    epoch_reward = 0 

    # Every 1000 epochs we have an episode
    if epoch % 1000 == 0: 
        print("Episode: " + str(epoch))

    while not done: 
        # Now we are in our gameloop
        # If some random number is greater than epsilon, then we take the best possible action we have explored so far
        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state])
        
        # Else, we will explore and take a random action
        else:

            action = np.random.randint(0, env.action_space.n) 

        # Now we will intialize our new_state, reward, and done variables
        new_state, reward, done, _ = env.step(action) 
    
        epoch_reward += reward 
        
        # We discretize our new state
        new_discrete_state = get_discrete_state(new_state)
        
        # We render our environment after 2000 steps
        if epoch % 2000 == 0: 
            env.render()

        # If the game loop is still running update the q-table
        if not done:
            max_new_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - lr) * current_q + lr * (reward + gamma* max_new_q)

            q_table[discrete_state + (action,)] = new_q
        discrete_state = new_discrete_state

    if epsilon > 0.05: 
        if epoch_reward > prev_reward and epoch > 10000:
            epsilon = math.pow(epsilon_decay_value, epoch - 10000)

            if epoch % 500 == 0:
                print("Epsilon: " + str(epsilon))

    #we calculate the final time
    tfinal = time.time() 
    
    #total epoch time
    episode_total = tfinal - t_initial 
    total_time += episode_total
    
    #calculate and update rewards
    total_reward += epoch_reward
    prev_reward = epoch_reward

    #every 1000 episodes print the average time and the average reward
    if epoch % 1000 == 0: 
        mean = total_time / 1000
        print("Time Average: " + str(mean))
        total_time = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

env.close()
