import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

FIX_STARTEND = False
class sim_env(object):
    def __init__(self, dim, propobility):
        self.dim = dim
        self.propobility = propobility
        self.WALL_VALUE = 50
        self.CAR_VALUE = 100
        self.GOAL_VALUE = 200
        self.max_step = 40

        self.state_dim = [(self.dim + 2) , (self.dim + 2)]
        self.action_dim = 4

    def reset(self):
      # np.random.seed(3)
      self.map_matrix = np.zeros([self.dim, self.dim])
      for i in range(self.dim):
        for j in range(self.dim):
          a = np.random.random(1)
          if a < self.propobility:
            self.map_matrix[i,j] = self.WALL_VALUE
          # pass

      # random start
      if FIX_STARTEND:
          self.start = 0, 0
      else:
          self.start = np.random.random_integers(0, self.dim-1, 2)
          self.start = self.start[0], self.start[1]
      self.map_matrix[self.start] = 0

      self.car_location = self.start

      # random goal
      if FIX_STARTEND:
          self.goal = self.dim-5, self.dim-5
      else:
          self.goal = np.random.random_integers(0, self.dim-1, 2)
          self.goal = self.goal[0], self.goal[1]
      self.map_matrix[self.goal] = self.GOAL_VALUE
      self.current_step = 0

      env_distance = 1 # env use car as center, sensing distance
      map_env = np.pad(self.map_matrix, env_distance,'constant', constant_values=self.WALL_VALUE)
      map_env[self.car_location[0] + 1, self.car_location[1] + 1] = self.CAR_VALUE
      return map_env


    # def plot_map(map_matrix, self.car_location):
    #     map_matrix[self.car_location] = CAR_VALUE# use three to present car
    #     plt.imshow(map_matrix, interpolation='none')

    def step(self, action):
        self.current_step += 1
        self.done = False

        feedback = 0 # default feedback 
        # env =  np.zeros([10, 10])	
        env_distance = 1 # env use car as center, sensing distance
        map_env = np.pad(self.map_matrix, env_distance,'constant', constant_values=self.WALL_VALUE)
        # map_env = np.copy(self.map_matrix)
        # if action == None:
        #     self.car_location = self.start
        #     car_x, car_y = self.car_location
        #     env_x = car_x + env_distance
        #     env_y = car_y + env_distance
        #     # env use car as center, sensing distance
        #     map_env[env_x, env_y] = CAR_VALUE
        #     # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        #     return self.car_location, feedback, map_env

        # check if initial_location legal
        if self.map_matrix[self.car_location] == self.WALL_VALUE:
            status = "initial position error"
            print "initial position error"
            # print("check car loc", self.car_location)
            # print(self.map_matrix)
            self.car_location = self.start
            self.done = True
            return map_env, feedback, self.done, status

        # do action, move the car
        car_x, car_y = self.car_location

        if action == 0:
            car_x -= 1
        elif action == 1:
            car_x += 1
        elif action == 2:
            car_y += 1
        elif action == 3:
            car_y -= 1
        else:
            print('action error!')
    	
        self.car_location = car_x, car_y

        env_x = car_x + env_distance
        env_y = car_y + env_distance
        env_location = env_x, env_y
        # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]
        goal_distance = np.sqrt(np.sum((np.asarray(self.goal) - np.asarray(self.car_location))**2)) # the distance from goal
        # print "goal_distance: ", goal_distance

        # print "step: ", step
        # check status
        status = 'normal'
        if map_env[env_location] == self.WALL_VALUE:
            # print "collision"
            feedback = -1 # collision feedback
            self.done = True
            status = 'collision'
            # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

        elif map_env[env_location] == 0:
            # improve = last_goaldistance - goal_distance # whether approach goal
            # if improve > 0:
            #     feedback = 0.001 # good moving feedback
            # elif improve < 0:
            #     feedback = -0.002 # bad moving feedback
            if self.current_step >= self.max_step:
                feedback = -1
                self.done = True
                status = 'max_step'

        elif map_env[env_location] == self.GOAL_VALUE:
            # print "congratulations! You arrive destination"
            feedback = 1 # get goal feedback
            self.done = True
            status = 'arrive'
        map_env[env_location] = self.CAR_VALUE

        # map_env = map_env.ravel
        # env = map_env[env_x - env_distance:env_x + env_distance + 1, env_y - env_distance: env_y + env_distance + 1]

        return map_env, feedback, self.done, status