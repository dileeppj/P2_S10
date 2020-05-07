import cv2
import time
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty,BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from ai import TD3

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

brain = TD3((1,40,40),1,5)
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0
    global Done
    Done = 0
    global count 
    count = 0
    global episode_timesteps
    episode_timesteps = 0

last_distance = 0

# Class for the autonomous car
class Car(Widget):
    
    angle = BoundedNumericProperty(0)
    rotation = BoundedNumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# Class for the game to be played by car in the app
class Game(Widget):
    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global Done
        global count
        global episode_timesteps
        
        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        image = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30:int(self.car.y)+30]
        
        image = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
        
        count += 1
        last_signal = image
        last_signal1 = [orientation, -orientation]
        action, episode_timesteps = brain.update(last_reward, last_signal,last_signal1, Done, count, episode_timesteps)
        Done = 0
        scores.append(brain.score())
        rotation = action
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            
            last_reward = -10
        else:
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.3
            else:
                last_reward = last_reward +(-0.2)

        if self.car.x < 30:
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)
            last_reward = -10
            Done = 1
        if self.car.x > self.width - 30:
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)
            last_reward = -10
            Done = 1
        if self.car.y < 30:
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)
            last_reward = -10
            Done = 1
        if self.car.y > self.height - 30:
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)
            last_reward = -10
            Done = 1
        
        if episode_timesteps > 2000:
            Done = 1
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)

        if distance < 25:
            Done = 1
            last_reward = 10
            self.car.x = randint(30, self.width)
            self.car.y = randint(30, self.height)
            if swap == 0:
                goal_x = 580
                goal_y = 530
                swap = 1
            elif swap == 1:
                goal_x = 1100
                goal_y = 310
                swap = 0
        last_distance = distance

# Class for autonomous car application
class CarApp(App):
    def build(self):
        game = Game()
        game.serve_car()
        Clock.schedule_interval(game.update, 1.0/60.0)
        return game

# Start point of the application
if __name__ == '__main__':
    CarApp().run()
