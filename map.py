#!/usr/bin/env python3
"""
Autonomuous car using TD3
* Using CNN for image based environment sensing.
* 
"""

# Imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, BoundedNumericProperty, NumericProperty, ReferenceListProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.config import Config
import numpy as np
from PIL import Image as PILImage

# import autoCarAI as brain
import ai as brain

# Setting the configuration for the project
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Class for the autonomous car
class AutoCar(Widget):
    # Angle and rotation of the car
    angle = BoundedNumericProperty(0, min=0, max=360,
    errorhandler=lambda x: 360 if x > 360 else 0)
    rotation = BoundedNumericProperty(0)
    # Velocity of the car in X and Y axis
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    # ReferenceList property; so we can use car.velocity 
    # as a shorthand just like: w.pos for w.x, w.y
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    def move(self, rotation):
        # print("Rotation :",rotation)
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# Class for the game to be played by car in the app
class AutoGame(Widget):
    car = ObjectProperty(None)
    first_update = True
    img = PILImage.open("./images/mask.png").convert('L')
    sand = (np.asarray(img)/255).astype(int)
    max_x, max_y = sand.shape
    position = Vector(int(max_x/2), int(max_y/2))
    angle = Vector(0,0).angle(position)
    max_angle = 10
    velocity = Vector(0,0)
    goal_x = 0
    goal_y = 0
    goal_iter = 0
    goals = [Vector(1040, 60), Vector(360, 340)]
    wall_padding = 10 # 10px padding on the wall
    done = True
    observation_space = 0
    action_space = 0
    _max_episode_steps = 0
    kiv_x = 0
    kiv_y = 0
    
    distance = 0
    scores = []
    
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    
    action_dim = 1
    state_dim = 40
    max_action = 10
    
    """## We set the parameters"""
    env_name = "AutonomousCar_TD3-v0" # Name of a environment (set it to any Continous environment you want)
    seed = 0 # Random seed number
    start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

    
    # We create the policy network (the Actor model)
    policy = brain.TD3(state_dim, action_dim, max_action)
    # We create the Experience Replay memory
    replay_buffer = brain.ReplayBuffer()


    # This methods resets the vehicles position, orientatin and velocity 
    def reset(self):
        self.car.x = np.random.randint(0,self.max_x)
        self.car.y = np.random.randint(0,self.max_y)
        self.car.angle = np.random.randint(0,360)
        self.car.velocity = Vector(2, 0).rotate(self.angle)
        print("[RESET  ] [ENV-RND_LOC ]Car position",self.car.pos,"Angle",self.car.angle,"degree")
        return self.get_state()

    # This method makes the car take random action in the action space of environment
    def random_action(self):
        rotation =  np.random.randint(low=-self.max_angle, high=self.max_angle)
        # print("Action :",rotation)
        return (rotation,)
    
    # def init_env(self):
    #     self.reset()
    #     print("[INFO   ] [Game        ]Car served at",self.car.pos,"at an angle of",self.car.angle,"degree")
    #     pass
    
    def serve_car(self):
        self.car.center = self.center # Center of car.
        self.car.velocity = Vector(0,0) # Velocity of car.
    
    def get_state(self):
        # This method will provide the current state of the vehicle,
        # we have to crop the portion of the loaction in map where the 
        # car is located.
        self.kiv_x = self.car.x
        self.kiv_y = self.max_y - self.car.y
        
        PIL_sand = PILImage.fromarray(self.sand)
        PIL_sand.convert("RGB").save("thumbnail_sand.jpg")
        
        area = (self.kiv_x - 80, self.kiv_y - 80, self.kiv_x + 80, self.kiv_y + 80)
        curr_state = self.img.rotate(90, expand=True).crop(area)
        curr_state.thumbnail((self.state_dim,self.state_dim))
        # print(area)
        curr_state.save("thumbnail.jpg","JPEG")
        return curr_state
    
    def step(self,action):
        # This method will correspond to a step (action) taken by the 
        # car in the environment.
        rotation = action[0]
        self.car.angle += rotation
        # print("cur Angle :",self.car.angle)
        self.car.pos = Vector(*self.car.velocity) + self.car.pos
        self.car.x = self.car.pos[0]
        self.car.y = self.car.pos[1]
        self.car.velocity = Vector(1, 0).rotate(self.car.angle)
        self.reward = 0
        self.done = False
        self.current_goal = self.goals[self.goal_iter]
        return self.get_state(),self.reward,self.done
    
    # def seed(self,seed):
    #     # This method sets the seed value for the environment 
    #     # randomness selection
    #     pass
    
    def update(self,dt):
        if self.first_update:
            # print("First Update")
            # self.init_env()
            obs = self.get_state()
            obs.save("thumbnail.jpg","JPEG")
            self.first_update = False
        # print("Update")
        # action = self.random_action()
        # obs = self.step(action)
        
        # Training
        self.total_timesteps = 0
        self.max_timesteps = 500000
        # We start the main loop over 500,000 timesteps
        if self.total_timesteps < self.max_timesteps:
            # If the episode is done
            if self.done:
                # If we are not at the very beginning, we start the training process of the model
                if self.total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
                    self.policy.train(self.replay_buffer, self.episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)
                    
                # When the training step is done, we reset the state of the environment
                obs = self.reset()

                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_num += 1
            
            # Before 10000 timesteps, we play random actions
            if self.total_timesteps < self.start_timesteps:
                action = self.random_action()
                obs = self.get_state()
            else: # After 10000 timesteps, we switch to the model
                action = self.policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
            
            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done = self.step(action)
            
            # Finding new Distance
            curr_pos = Vector(self.car.pos)
            new_distance = curr_pos.distance(self.current_goal)
            print(new_distance)
            
            # Adding environment conditions
            if self.sand[int(self.car.x),int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                reward = -0.5
                self.done = False
            else:
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                reward = 10
                self.done = False
                if new_distance < self.distance:
                    reward = 20

            if self.car.x < 20:
                self.car.x = 20
                reward -= 10
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward -= 10
                self.done = True
            if self.car.y < 10:
                self.car.y = 10
                reward -= 10
                self.done = True
            if self.car.y > self.height - 10:
                self.car.y = self.height - 10
                reward -= 10
                self.done = True

            if new_distance < 25:
                reward += 35
                if swap == 1:
                    print("Reached position A: x: ",self.car.x," y: ",self.car.y)
                    self.goal_iter = 1
                    swap = 0
                    self.done = True
                else:
                    print("Reached position B: x: ",self.car.x," y: ",self.car.y)
                    self.goal_iter = 0
                    swap = 1
                    self.done = False
            else:
                reward -= 0.1

            self.scores.append(self.policy.score())

            # We check if the episode is done
            done_bool = 0 if self.episode_timesteps + 1 == self._max_episode_steps else float(done)
                
            # We increase the total reward
            self.episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            self.replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            self.distance = new_distance
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1
                
                
# Class for autonomous car application
class autoCarApp(App):
    def build(self):
        game = AutoGame()
        # Serve the car with a velocity of 6,0 in x,y
        game.serve_car()
        # Update the move function 60 times a second
        Clock.schedule_interval(game.update, 1.0/60.0)
        return game

# Start point of the application
if __name__ == "__main__":
    carApp = autoCarApp()
    carApp.run()