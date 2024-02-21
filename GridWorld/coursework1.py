#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import random
import math
import matplotlib.pyplot as plt # Graphical library

from sklearn.metrics import mean_squared_error # Mean-squared error function


# # Coursework 1 :
# See pdf for instructions. 

# In[2]:


# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]

def get_BTcode():
    return "bt723769" # Return your btcode in the format 'bt######'

def get_fullName():
    return "Ivan Khrop" # Return your full name


# ## Helper class

# In[3]:


# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework

class GraphicsMaze(object):

    def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

        self.shape = shape
        self.locations = locations
        self.absorbing = absorbing

        # Walls
        self.walls = np.zeros(self.shape)
        for ob in obstacle_locs:
            self.walls[ob] = 20

        # Rewards
        self.rewarders = np.ones(self.shape) * default_reward
        for i, rew in enumerate(absorbing_locs):
            self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

        # Print the map to show it
        self.paint_maps()

    def paint_maps(self):
        """
        Print the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        plt.figure(figsize=(15,10))
        plt.imshow(self.walls + self.rewarders)
        plt.show()

    def paint_state(self, state):
        """
        Print one state on the Maze topology (obstacles, absorbing states and rewards)
        input: /
        output: /
        """
        states = np.zeros(self.shape)
        states[state] = 30
        plt.figure(figsize=(15,10))
        plt.imshow(self.walls + self.rewarders + states)
        plt.show()

    def draw_deterministic_policy(self, Policy):
        """
        Draw a deterministic policy
        input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(15,10))
        plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
        for state, action in enumerate(Policy):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
            action_arrow = arrows[action] # Take the corresponding action
            location = self.locations[state] # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
        plt.show()

    def draw_policy(self, Policy):
        """
        Draw a policy (draw an arrow in the most probable direction)
        input: Policy {np.array} -- policy to draw as probability
        output: /
        """
        deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
        self.draw_deterministic_policy(deterministic_policy)

    def draw_value(self, Value):
        """
        Draw a policy value
        input: Value {np.array} -- policy values to draw
        output: /
        """
        plt.figure(figsize=(15,10))
        plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
        for state, value in enumerate(Value):
            if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
                continue
            location = self.locations[state] # Compute the value location on graph
            plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
        plt.show()

    def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple deterministic policies
        input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
        output: /
        """
        plt.figure(figsize=(20,8))
        for subplot in range (len(Policies)): # Go through all policies
            ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
            ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
            for state, action in enumerate(Policies[subplot]):
                if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                    continue
                # List of arrows corresponding to each possible action
                arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
                action_arrow = arrows[action] # Take the corresponding action
                location = self.locations[state] # Compute its location on graph
                plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
            ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
        plt.show()

    def draw_policy_grid(self, Policies, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policies (draw an arrow in the most probable direction)
        input: Policy {np.array} -- array of policies to draw as probability
        output: /
        """
        deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
        self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

    def draw_value_grid(self, Values, title, n_columns, n_lines):
        """
        Draw a grid representing multiple policy values
        input: Values {np.array of np.array} -- array of policy values to draw
        output: /
        """
        plt.figure(figsize=(20,8))
        for subplot in range (len(Values)): # Go through all values
            ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
            ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
            for state, value in enumerate(Values[subplot]):
                if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                    continue
                location = self.locations[state] # Compute the value location on graph
                plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
            ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.show()


# ## Maze class

# In[4]:


# This class define the Maze environment

class Maze(object):

    # [Action required]
    def __init__(self, y=6, z=9):
        """
        Maze initialisation.
        input: /
        output: /
        """

        # [Action required]
        # Properties set from the CID
        self._prob_success = 0.8 + 0.02 * (9 - y) # float
        self._gamma = 0.8 + 0.02 * y # float
        self._goal = z % 4 # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

        # Build the maze
        self._build_maze()


    # Functions used to build the Maze environment 
    # You DO NOT NEED to modify them
    def _build_maze(self):
        """
        Maze initialisation.
        input: /
        output: /
        """

        # Properties of the maze
        self._shape = (13, 10)
        self._obstacle_locs = [
                              (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                              (2,1), (2,2), (2,3), (2,7), \
                              (3,1), (3,2), (3,3), (3,7), \
                              (4,1), (4,7), \
                              (5,1), (5,7), \
                              (6,5), (6,6), (6,7), \
                              (8,0), \
                              (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                              (10,0)
                             ] # Location of obstacles
        self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
        self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
        self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
        self._default_reward = -1 # Reward for each action performs in the environment
        self._max_t = 500 # Max number of steps in the environment

        # Actions
        self._action_size = 4
        self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on

        # States
        self._locations = []
        for i in range (self._shape[0]):
            for j in range (self._shape[1]):
                loc = (i,j) 
                # Adding the state to locations if it is no obstacle
                if self._is_location(loc):
                    self._locations.append(loc)
        self._state_size = len(self._locations)

        # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
        self._neighbours = np.zeros((self._state_size, 4)) 

        for state in range(self._state_size):
            loc = self._get_loc_from_state(state)

            # North
            neighbour = (loc[0]-1, loc[1]) # North neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
            else: # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index('N')] = state

            # East
            neighbour = (loc[0], loc[1]+1) # East neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
            else: # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index('E')] = state

            # South
            neighbour = (loc[0]+1, loc[1]) # South neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
            else: # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index('S')] = state

            # West
            neighbour = (loc[0], loc[1]-1) # West neighbours location
            if self._is_location(neighbour):
                self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
            else: # If there is no neighbour in this direction, coming back to current state
                self._neighbours[state][self._direction_names.index('W')] = state

            # Absorbing
            self._absorbing = np.zeros((1, self._state_size))
            for a in self._absorbing_locs:
                absorbing_state = self._get_state_from_loc(a)
                self._absorbing[0, absorbing_state] = 1

            # Transition matrix
            
            self._T = self._fill_in_transition()
            
            """
            self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
            for action in range(self._action_size):
                for outcome in range(4): # For each direction (N, E, S, W)
                    # The agent has prob_success probability to go in the correct direction
                    if action == outcome:
                        # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
                        prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) 
                    # Equal probability to go into one of the other directions
                    else:
                        prob = (1.0 - self._prob_success) / 3.0

                # Write this probability in the transition matrix
                for prior_state in range(self._state_size):
                  # If absorbing state, probability of 0 to go to any other states
                  if not self._absorbing[0, prior_state]:
                    post_state = self._neighbours[prior_state, outcome] # Post state number
                    post_state = int(post_state) # Transform in integer to avoid error
                    self._T[prior_state, post_state, action] += prob
            """
            # Reward matrix
            
            self._R = self._fill_in_reward()
            
            """
            self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
            self._R = self._default_reward * self._R # Set default_reward everywhere
            
            for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
                post_state = self._get_state_from_loc(self._absorbing_locs[i])
                self._R[:,post_state,:] = self._absorbing_rewards[i]
            """

        # Creating the graphical Maze world
        self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)

        # Reset the environment
        self.reset()
        
    def _fill_probability_for_direction(self, target, to_go, state):

        probabilities = np.zeros(self._state_size, dtype=float)

        # number of ways from state
        n_ways = len(list(filter(lambda key: key != target and to_go[key] is not None, to_go)))

        for key in to_go:
            direction, next_state = key, to_go[key]

            ## i can go in the direction and change state
            if target == direction and next_state is not None:
                probabilities[next_state] = self._prob_success
            ## i try go in the direction but it's not possible => i stay on place
            elif target == direction and next_state is None:
                probabilities[state] = self._prob_success
            ## in case that i fail and go other direction
            elif target != direction and next_state is not None:
                probabilities[next_state] = (1 - self._prob_success) / n_ways if n_ways != 0 else 0.0

        ## in case of some changes !!! for this example is always 0.0
        probabilities[state] += 1.0 - np.sum(probabilities)

        return probabilities
    
    def _fill_in_transition(self):
        """
        Compute the transition matrix of the grid
        input: /
        output: T {np.array} -- the transition matrix of the grid
        """
        T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of dimension S*S*A

        ####
        # Add your code here
        # Hint! You might need: self.action_size, self.state_size, self.neighbours, self.prob_success
        ####

        for i in range(self._shape[0]):
            for j in range(self._shape[1]):

                ## if we are not in the state, just go skip
                if not self._is_location((i, j)):
                    continue

                ## we work with a state
                curr_state = self._get_state_from_loc((i, j))

                ## if it's absorbing state => stay here !!!
                if self._is_absorbing((i, j)):
                    T[(curr_state, curr_state)] = np.ones(self._action_size, dtype=float)
                    continue

                ## save directions we can go ---- use function is_location
                to_go = {}
                to_go['north'] = self._get_state_from_loc((i - 1, j)) if self._is_location((i - 1, j)) else None
                to_go['east'] = self._get_state_from_loc((i, j + 1)) if self._is_location((i, j + 1)) else None
                to_go['south'] = self._get_state_from_loc((i + 1, j)) if self._is_location((i + 1, j)) else None
                to_go['west'] = self._get_state_from_loc((i, j - 1)) if self._is_location((i, j - 1)) else None

                ## unique transition matrix for each action !!!
                T[curr_state,:,0] = self._fill_probability_for_direction('north', to_go, curr_state)
                T[curr_state,:,1] = self._fill_probability_for_direction('east', to_go, curr_state)
                T[curr_state,:,2] = self._fill_probability_for_direction('south', to_go, curr_state)
                T[curr_state,:,3] = self._fill_probability_for_direction('west', to_go, curr_state)

        return T
    
    def _fill_in_reward(self):
        """
        Compute the reward matrix of the grid
        input: /
        output: R {np.array} -- the reward matrix of the grid
        """
        R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
        R = self._default_reward * R # Set default_reward everywhere
        ####
        # Add your code here
        # Hint! You might need: self.absorbing_rewards, self.absorbing_locs, self.get_state_from_loc()
        ####

        for loc, reward in zip(self._absorbing_locs, self._absorbing_rewards):

            state = self._get_state_from_loc(loc)
            R[state, state, :] = reward

            # if in the north we have a simple state, then going south leads to an absorbing state
            if self._is_location((loc[0] - 1, loc[1])) and not self._is_absorbing((loc[0] - 1, loc[1])):
                source_state = self._get_state_from_loc((loc[0] - 1, loc[1]))
                R[source_state, state, 2] = reward

            # if in the south we have a simple state, then going north leads to an absorbing state
            if self._is_location((loc[0] + 1, loc[1])) and not self._is_absorbing((loc[0] + 1, loc[1])):
                source_state = self._get_state_from_loc((loc[0] + 1, loc[1]))
                R[source_state, state, 0] = reward

            # if in the east we have a simple state, then west south leads to an absorbing state
            if self._is_location((loc[0], loc[1] + 1)) and not self._is_absorbing((loc[0], loc[1] + 1)):
                source_state = self._get_state_from_loc((loc[0], loc[1] + 1))
                R[source_state, state, 3] = reward

            # if in the west we have a simple state, then east south leads to an absorbing state
            if self._is_location((loc[0], loc[1] - 1)) and not self._is_absorbing((loc[0], loc[1] - 1)):
                source_state = self._get_state_from_loc((loc[0], loc[1] - 1))
                R[source_state, state, 1] = reward

        return R


    def _is_location(self, loc):
        """
        Is the location a valid state (not out of Maze and not an obstacle)
        input: loc {tuple} -- location of the state
        output: _ {bool} -- is the location a valid state
        """
        if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
            return False
        elif (loc in self._obstacle_locs):
            return False
        else:
            return True
        
        
    def _is_absorbing(self, loc):
        """
        Is the location an absorbing state
        input: loc {tuple} -- location of the state
        output: _ {bool} -- is the location an absorbing state
        """
        return self._is_location(loc) and (loc in self._absorbing_locs)
    

    def _get_state_from_loc(self, loc):
        """
        Get the state number corresponding to a given location
        input: loc {tuple} -- location of the state
        output: index {int} -- corresponding state number
        """
        return self._locations.index(tuple(loc))


    def _get_loc_from_state(self, state):
        """
        Get the state number corresponding to a given location
        input: index {int} -- state number
        output: loc {tuple} -- corresponding location
        """
        return self._locations[state]

    # Getter functions used only for DP agents
    # You DO NOT NEED to modify them
    def get_T(self):
        return self._T

    def get_R(self):
        return self._R

    def get_absorbing(self):
        return self._absorbing

    # Getter functions used for DP, MC and TD agents
    # You DO NOT NEED to modify them
    def get_graphics(self):
        return self._graphics

    def get_action_size(self):
        return self._action_size

    def get_state_size(self):
        return self._state_size

    def get_gamma(self):
        return self._gamma

    # Functions used to perform episodes in the Maze environment
    def reset(self):
        """
        Reset the environment state to one of the possible starting states
        input: /
        output: 
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """
        self._t = 0
        self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
        self._reward = 0
        self._done = False
        return self._t, self._state, self._reward, self._done

    def step(self, action):
        """
        Perform an action in the environment
        input: action {int} -- action to perform
        output: 
          - t {int} -- current timestep
          - state {int} -- current state of the envionment
          - reward {int} -- current reward
          - done {bool} -- True if reach a terminal state / 0 otherwise
        """

        # If environment already finished, print an error
        if self._done or self._absorbing[0, self._state]:
            print("Please reset the environment")
            return self._t, self._state, self._reward, self._done

        # Drawing a random number used for probaility of next state
        probability_success = random.uniform(0,1)

        # Look for the first possible next states (so get a reachable state even if probability_success = 0)
        new_state = 0
        while self._T[self._state, new_state, action] == 0: 
            new_state += 1
        assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

        # Find the first state for which probability of occurence matches the random value
        total_probability = self._T[self._state, new_state, action]
        while (total_probability < probability_success) and (new_state < self._state_size-1):
            new_state += 1
            total_probability += self._T[self._state, new_state, action]
        assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

        # Setting new t, state, reward and done
        self._t += 1
        self._reward = self._R[self._state, new_state, action]
        self._done = self._absorbing[0, new_state] or self._t > self._max_t
        self._state = new_state
        return self._t, self._state, self._reward, self._done


# ## DP Agent

# In[5]:


# This class define the Dynamic Programing agent 

class DP_agent(object):

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env, threshold=1e-5):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - V {np.array} -- Corresponding value function 
        """

        # Initialisation (can be edited)
        V = np.zeros(env.get_state_size())

        #### 
        # Add your code here
        # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
        ####
        
        self.transitions = np.transpose(env.get_T(), (2, 0, 1))
        self.rewards = np.transpose(env.get_R(), (2, 0, 1))
        
        self.absorbing = np.squeeze(np.array(env.get_absorbing(), dtype=bool))
        self.gamma = env.get_gamma()
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        
        delta = 2 * threshold
        epochs = 0
        
        while delta > threshold:
            epochs += 1
            # calculate q_values
            q_values = self.estimate_q_values(V)
            
            # update value function according to the best action in time
            new_values = np.max(q_values, axis=1)
            new_values[self.absorbing] = 0.0
            
            # next iteration
            delta = np.linalg.norm(new_values - V)
            V = new_values
        
        q_values = self.estimate_q_values(V)
        # identify actions with the best state-action value function
        policy = np.array(abs(np.max(q_values, axis=1, keepdims=True) - q_values) <= 1e-12, dtype=float)
        # in case of several best actions. In this case each action is equally probable
        policy /= np.sum(policy, axis=1).reshape(-1, 1)

        return policy, V
    
    ## calculate q_values
    def estimate_q_values(self, V):
        q_values = self.transitions * (self.gamma * V.reshape(1, 1, self.state_size) + self.rewards)
        q_values = np.sum(q_values, axis=2).T

        return q_values


# In[6]:


## sequences for future agents

## learning rate generator        
def base_sequence():
    k = 1
    
    while True:
        n_log = math.trunc(math.log(k, 10.0)) - 2.0
        yield 1.0 / (10.0 ** n_log) if n_log > 1 else 0.01
        k += 1        
    
## e-greedy argument generator (explore >>>> exploit)
def eps_sequence(base=0.995):
    eps = 1.0
    while True:
        yield eps
        eps *= base


# ## MC agent

# In[7]:


# This class define the Monte-Carlo agent

class MC_agent(object):
      
    # basic setup
    # possible modes = 'learning_rate', 'first_visit', 'each_visit'
    def __init__(self, mode='learning_rate', base=0.9995):
        
        self.base = base
        
        ## just for testing
        self.mode = mode
    
    def __str__(self):
        return 'MC-Agent. Learning mode = {}.'.format(self.mode)
    
    # returns e-greedy policy
    def _eps_greedy_policy(self, eps):
        
        # get best actions ==> mask has True on the best actions for each state
        mask = abs(np.max(self.Q, axis=1, keepdims=True) - self.Q) <= 1e-7
        
        # create new policy like e-greedy
        new_policy = np.zeros(mask.shape)
        new_policy[mask] = (1 - eps)  
        new_policy /= np.sum(mask, axis=1).reshape(-1, 1)
        
        
        new_policy[~mask] = eps / self.n_act
        new_policy[mask] += eps / self.n_act
        
        return new_policy 
    
    # calculates backward-rewards for all episodes
    def _estimate(self, episodes):
        
        ## to calculate average total reward for batch
        total_reward = 0.0
        average_length = 0.0
        
        ## go over all episodes
        for idx, episode in enumerate(episodes):
            returns = np.zeros(len(episode), dtype=float)
            average_length += len(episode) / len(episodes)
            
            ## go backward and calculate returns for each step in episode
            for i, elem in enumerate(episode[len(episode)-2::-1]):
                returns[i + 1] = elem[2] + self.discount * returns[i]
                total_reward += elem[2] / len(episodes)
                
            returns = returns[::-1]
    
            ## estimate the results and update values
            self._visit_estimation(episode, returns)
        
            ## save the obtained value function and average reward for batch
            if idx == len(episodes) - 1:
                self.values.append(np.copy(self.V))
                self.episode_lengths.append(average_length)
                self.total_rewards.append(total_reward)
        
    def _visit_estimation(self, trace, returns):
        
        state_visit = np.zeros(self.n_stat, dtype=bool)
        state_act_visit = np.zeros((self.n_stat, self.n_act), dtype=bool)
        
        lr = next(self.lr) if self.mode == 'learning_rate' else None
        
        ## for each step in trace
        for step, ret in zip(trace, returns):
            state, act = step[0], step[1]
            
            if act == None:
                continue
            
            # if first_visit => then it will be used only ones
            # just for state-value function
            if not state_visit[state] or self.mode != 'first_visit':
                self.value_counter[state] += 1
                
                coeff = lr if lr is not None else 1.0 / self.value_counter[state]
                
                self.V[state] += coeff * (ret - self.V[state])
                state_visit[state] = True
            
            # if first_visit => then it will be used only ones 
            # just for state-action-value function
            if not state_act_visit[state, act] or self.mode != 'first_visit':
                self.q_value_counter[state, act] += 1
                
                coeff = lr if lr is not None else 1.0 / self.q_value_counter[state, act]
                
                self.Q[state, act] += coeff * (ret - self.Q[state, act])
                state_act_visit[state, act] = True
            
            # if we already seen all state_action combinations (do we need it ????)
            if state_act_visit.all() and self.mode == 'first_visit':
                break
        
  
    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env, batch_size=1):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        
        ## sequences for learning_rate and eps_greedy values
        self.lr = base_sequence()
        self.eps = eps_sequence(self.base)

        # Initialisation (can be edited)
        self.n_stat = int(env.get_state_size())
        self.n_act = int(env.get_action_size())
        self.discount = env.get_gamma()

        # value functions and counters
        self.Q = np.zeros((self.n_stat, self.n_act))
        self.q_value_counter = np.zeros((self.n_stat, self.n_act), dtype=int)
        self.V = np.zeros(self.n_stat)
        self.value_counter = np.zeros(self.n_stat, dtype=int)

        # results
        self.values = [np.copy(self.V)]
        self.total_rewards = []
        self.episode_lengths = []

        #### 
        # Add your code here
        # WARNING: this agent only has access to env.reset() and env.step()
        # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
        ####
        
        print('Progress: ', end='')
        
        n_steps = math.ceil(math.log(0.01, self.base) / 1_000) * 1_000
        
        for i in range(n_steps):
            # get next eps-value from generator
            eps = next(self.eps)
            # set e-greedy policy
            e_greedy_policy = self._eps_greedy_policy(eps)
            
            # run model batch_size-times to recalculate Q(s,a) and V(s)
            episodes = []
            for k in range(batch_size):
                
                trace = []
                t, state, reward, done = env.reset() 
                
                while not done:
                    # choose an action
                    action = np.random.choice(a=np.arange(self.n_act), p=e_greedy_policy[state])
                    # next state using the chosen action
                    t, next_state, reward, done = env.step(action)
                    # add data into a trace
                    trace.append((state, action, reward))
                    # save state for next step
                    state = next_state
                    
                # technical addition to trace
                trace.append((state, None, None))
                
                episodes.append(trace)
            
            self._estimate(episodes)
            
            if i % (n_steps // 10) == 0:
                print('{}% '.format(round(i * 100 / n_steps)), end='')
                
        print('100%')

        return self._eps_greedy_policy(0.0), self.values, self.total_rewards


# ## TD agent

# In[8]:


# This class define the Temporal-Difference agent

class TD_agent(object):

    ## two possible modes: 'off-policy', 'on-policy'
    def __init__(self, mode='on-policy', base=0.9995):      
        
        self.base = base
        
        ## save mode for testing
        self.mode = mode
        
    def __str__(self):
        return 'TD-Agent. Learning mode = {}.'.format(self.mode)
        
     # returns e-greedy policy
    def _eps_greedy_policy(self, eps):
        
        # get best actions ==> mask has True on the best actions for each state
        mask = abs(np.max(self.Q, axis=1, keepdims=True) - self.Q) <= 1e-7
        
        # create new policy like e-greedy
        new_policy = np.zeros(mask.shape)
        new_policy[mask] = (1 - eps)  
        new_policy /= np.sum(mask, axis=1).reshape(-1, 1)
        
        
        new_policy[~mask] = eps / self.n_act
        new_policy[mask] += eps / self.n_act
        
        return new_policy
    
    def _estimate(self, old_state, new_state, old_action, new_action, reward, lr):
                        
        # update value function by temporal difference
        update_value = (reward + self.discount * self.V[new_state] - self.V[old_state])
        self.V[old_state] += lr * update_value
        
        ## update state-action value function
        if self.mode == 'off-policy':
            # off-policy => using the best action at the next step
            update_value = reward + self.discount * np.max(self.Q[new_state]) - self.Q[old_state, old_action]
            self.Q[old_state, old_action] += lr * update_value
                
        else:
            # on-policy => using just state-acion value of my future step
            update_value = reward + self.discount * self.Q[new_state, new_action] - self.Q[old_state, old_action]
            self.Q[old_state, old_action] += lr * update_value
    
    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """
        
        ## sequences for learning_rate and eps_greedy values
        self.lr = base_sequence()
        self.eps = eps_sequence(self.base)
        
        # Initialisation (can be edited)
        self.n_stat = int(env.get_state_size())
        self.n_act = int(env.get_action_size())
        self.discount = env.get_gamma()

        # value functions
        self.Q = np.zeros((self.n_stat, self.n_act)) 
        self.V = np.zeros(self.n_stat)
        self.values = [np.copy(self.V)]
        self.total_rewards = []
        self.episode_lengths = []

        #### 
        # Add your code here
        # WARNING: this agent only has access to env.reset() and env.step()
        # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
        ####
        
        print('Progress: ', end='')
        
        n_steps = math.ceil(math.log(0.01, self.base) / 1_000) * 1_000
        
        for i in range(n_steps):
                        
            # get next eps-value from generator
            eps = next(self.eps)
            # set e-greedy policy
            e_greedy_policy = self._eps_greedy_policy(eps)
            
            # run model batch_size-times to recalculate Q(s,a) and V(s)
            t, state, reward, done = env.reset()
            # choose an action
            action = np.random.choice(a=np.arange(self.n_act), p=e_greedy_policy[state])
            
            # total reward too save
            total_reward = 0.0
            lr = next(self.lr)
            
            while not done:
                # walk just one step and get the results of this step
                t, next_state, reward, done = env.step(action)
                next_action = np.random.choice(a=np.arange(self.n_act), p=e_greedy_policy[next_state])
                
                # update value function and state-action value function
                self._estimate(state, next_state, action, next_action, reward, lr)
                
                # for next interation
                state, action = next_state, next_action
                total_reward += reward

            # save data
            self.values.append(np.copy(self.V))
            self.total_rewards.append(total_reward)
            self.episode_lengths.append(t)
            
            if i % (n_steps // 10) == 0:
                print('{}% '.format(round(i * 100 / n_steps)), end='')
                
        print('100%')

        return self._eps_greedy_policy(0.0), self.values, self.total_rewards


# ## Some functions that will be used later

# In[9]:


## calculate error for one value-function
def calculate_error(predicted_values, true_value):
    """
    Calculates MSE for each predicted value 
    input: 
      - predicted_values {np.array shape=(n_predicts, n_episodes, n_values)} -- predicted results
      - true_value {np.array} -- true result
    output: 
      - error_values {np.array} -- MSE for each predict 
    """
    error_func = lambda value: mean_squared_error(true_value, value)
    return np.apply_along_axis(error_func, 2, predicted_values) 

## results regarding total reward
def agent_evaluation(n_replication, agent, env, true_values):
    
    """
    Calculates mean MSE for an agent 
    input: 
      - n_replication {int} -- number of runs of agent
      - agent {MC_agent or TD_agent} - agent that will be run
      - env {Maze object} -- Maze to solve
      - true_value {np.array} -- true result from DP_agent run
    output: 
      - mean_total_reward {np.array} -- mean total reward of agent through learning
      - std_total_reward {np.array} -- STD for total reward of agent through learning
      - mean_error {np.array} -- mean MSE of agent
      - std_error {np.array} -- STD for MSE of agent
    """
    
    data_reward = None
    data_values = []
    
    print(agent)

    for i in range(n_replication):
        print('Evaluation run #{}.'.format(i + 1))
        print('==' * 60)
        
        # run agent
        policy, values, total_rewards = agent.solve(maze)
        
        # save results
        data_values.append(np.array(values))
        if data_reward is None:
            data_reward = np.copy(np.array(total_rewards))
        else:
            data_reward = np.vstack([data_reward, np.array(total_rewards)])
            
        print('==' * 60)
        
    error = calculate_error(np.array(data_values), true_values)
        
    return np.mean(data_reward, axis=0), np.std(data_reward, axis=0), np.mean(error, axis=0), np.std(error, axis=0)


# ## Example main

# In[10]:


# Example main (can be edited)

### Question 0: Defining the environment

print("Creating the Maze:\n")
maze = Maze()


# ## Dynamic Programming Agent

# In[11]:


### Question 1: Dynamic programming

dp_agent = DP_agent()
dp_policy, dp_value = dp_agent.solve(maze)

print("Results of the DP agent:\n")
maze.get_graphics().draw_policy(dp_policy)
maze.get_graphics().draw_value(dp_value)


# ## Monte-Carlo Agent

# In[12]:


### Question 2: Monte-Carlo learning

mc_agent = MC_agent(mode='learning_rate', base=0.9995)
mc_policy, mc_values, total_rewards = mc_agent.solve(maze, batch_size=1)

print("Results of the MC agent:\n")
maze.get_graphics().draw_policy(mc_policy)
maze.get_graphics().draw_value(mc_values[-1])


# In[13]:


plt.plot(np.arange(1, len(total_rewards) + 1), total_rewards)
plt.xlabel('Number of iterations')
plt.ylabel('Undiscounted reward')
plt.show()


# In[14]:


n_replicas = 150
mc_agent = MC_agent(mode='learning_rate', base=0.9995)

mean_values, std_values, mc_MSE_mean, mc_MSE_std = agent_evaluation(n_replicas, mc_agent, maze, dp_value)


# In[15]:


# let's draw a plot
n_episodes = np.arange(1, len(mean_values) + 1)
plt.plot(n_episodes, mean_values, color='r', label='Mean Episode-Reward')
plt.fill_between(n_episodes, mean_values - std_values, mean_values + std_values, 
                 alpha=0.4, color='gray', label='STD Episode-Reward')
plt.xlabel('Number of episodes')
plt.ylabel('Undiscounted reward')
plt.legend(loc='lower right')
plt.title(label='Undiscounted reward dynamic for MC-Agent', loc='center')
plt.show()


# ## TD Agent

# In[25]:


td_agent = TD_agent(mode='on-policy', base=0.9995)
td_policy, td_values, total_rewards = td_agent.solve(maze)

print("Results of the TD agent:\n")
maze.get_graphics().draw_policy(td_policy)
maze.get_graphics().draw_value(td_values[-1])


# In[26]:


plt.plot(np.arange(1, len(total_rewards) + 1), total_rewards)
plt.xlabel('Number of iterations')
plt.ylabel('Undiscounted reward')
plt.show()


# In[18]:


n_replicas = 150
td_agent = TD_agent(mode='on-policy', base=0.9995)

mean_values, std_values, td_MSE_mean, td_MSE_std = agent_evaluation(n_replicas, td_agent, maze, dp_value)


# In[19]:


# let's draw a plot
n_episodes = np.arange(1, len(mean_values) + 1)
plt.plot(n_episodes, mean_values, color='r', label='Mean Episode-Reward')
plt.fill_between(n_episodes, mean_values - std_values, mean_values + std_values, 
                 alpha=0.4, color='gray', label='STD Episode-Reward')
plt.xlabel('Number of episodes')
plt.ylabel('Undiscounted reward')
plt.title(label='Undiscounted reward dynamic for TD-Agent', loc='center')
plt.legend(loc='lower right')
plt.show()


# ## Comparison with the results of DP Agent

# In[20]:


# print MC-Agent Error
plt.figure(figsize=(8, 8))

x_values = np.arange(1, len(mc_MSE_mean) + 1)
plt.plot(x_values, mc_MSE_mean, color='r', label='Mean MSE for MC-Agent')
plt.fill_between(x_values, mc_MSE_mean - mc_MSE_std, 
                 mc_MSE_mean + mc_MSE_std, alpha=0.3, color='r')

# print TD-Agent Error
x_values = np.arange(1, len(td_MSE_mean) + 1)
plt.plot(x_values, td_MSE_mean, color='b', label='Mean MSE for TD-Agent')
plt.fill_between(x_values, td_MSE_mean - td_MSE_std,
                 td_MSE_mean + td_MSE_std, alpha=0.3, color='b')

# rest of plot data
plt.xlabel('Number of episodes')
plt.ylabel('MSE Error rate')
plt.title(label='MSE dynamics for MC and TD agents', loc='center')
plt.legend(loc='best')
plt.show()


# In[21]:


# MC-Agent
mc_agent = MC_agent(mode='learning_rate', base=0.9995)
mc_policy, mc_values, mc_total_rewards = mc_agent.solve(maze)

mc_error = calculate_error(np.array([mc_values]), dp_value)

# TD-Agent
td_agent = TD_agent(mode='on-policy', base=0.9995)
td_policy, td_values, td_total_rewards = td_agent.solve(maze)

td_error = calculate_error(np.array([td_values]), dp_value)


# In[22]:


# draw scatter plots
plt.figure(figsize=(10, 10))
plt.scatter(mc_total_rewards, mc_error[0, 1:], edgecolors='r', label='MC-Agent', color='white')
plt.scatter(td_total_rewards, td_error[0, 1:], edgecolors='b', label='TD-Agent', color='white')

# rest of the plot
plt.xlabel('Undiscounter total reward')
plt.ylabel('MSE-value')
plt.title(label='MSE with respect to undiscounter reward', loc='center')
plt.legend()
plt.show()

