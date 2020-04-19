# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import cv2

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1503')
Config.set('graphics', 'height', '892')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

brain = TD3((1,60,60),1,5)
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))     # Longitude and Latitude [Width x Height]
    img = PILImage.open("./images/mask.png").convert('L')   # Convert to the lower values - we want one channel only [2D array]
    sand = np.asarray(img)/255
    # First Target to reach - Lake Park
    goal_x = 1050
    goal_y = 110
    first_update = False
    global toPos     # toPos the goal
    toPos = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    def move(self, rotation):
        # Move car and the sensor
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

# Creating the game class

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
        global toPos
        

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
        
        # xx, yy is used to peanilize the wrong action, as it shows the dist b/w car and goal. when new dist > old dist penilize
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        image = sand[int(self.car.x)-30:int(self.car.x)+30, int(self.car.y)-30:int(self.car.y)+30]
        image = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
        count += 1
        last_signal = image
        last_signal1 = [orientation, -orientation]
        action, episode_timesteps = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        # if it is at sand/ at illegal area
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            # It is the penalty for going to illegal area
            last_reward = -3
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # This reward is living penalty
            last_reward = -0.8
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                # If we are going towards goal, we get +ve goal
                last_reward = 1
            # else:
            #     last_reward = last_reward +(-0.2)

        # If car is going into wall
        if self.car.x < 5:
            self.car.x = 5
            last_reward = -3
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -3
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -3
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -3

        if distance < 50:
            # If we are 25 pixel near goal change the goals
            # TODO: 3 Tragets.
            if toPos == 0:
                goal_x = 900
                goal_y = 230
                toPos = 1
            elif toPos == 1:
                goal_x = 1050
                goal_y = 110
                toPos = 2
            else:
                goal_x = 260
                goal_y = 408
                toPos = 0
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
