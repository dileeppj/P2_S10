# P2_S10

## Autonomous Car using TD3 Reinforcement Learning

### Aim
Make an environment, Train the car to move from point A to point B and back through the road, following the shortest path with sensor data as image which we get from the environment. We have to use TD3 Reinforcement Learning with a basic CNN to get the sensory data.

### Implementation Details
We are using Kivy for simulating the environment, the environment will be providing feedback to the actions took by the car. The current state of the car is described by a 40x40 section of the location where the car is located. This becomes the state_dim for the TD3 algorithm. The TD3 algorithm will provide the action to be taken, which for our case is the angle of rotation for the car. The velocity of the car is determined by whether the car is on sand or the road. The image which we load in Kivy has the coordinate system different from that of PIL and numpy, the origin is located at bottom left for Kivy and top left for PIL and numpy.
The send the cropped image to the network where it predicts the next state and reward if we take the current step.

First, the car is made to run in exploration mode, where it takes random actions and learn those experiences. We will be taking 10000 random steps in the expolration mode. These are stored in the ReplayBuffer < curObs, curAct, Reward, nextObs >. The curObs is the current observation, curAct is the Action took for the curObs, nextObs is the scene which we will end up in after taking that step and Reward is the reward which we got for that action.
 Next, we make the actor to select action
 
 
 ### Explanation 
* car.kv

  The kv file which describes the rules which are used to describe the content of a Widget used in the game.
* map.py

  The main file where the App and Game are defined. The environment is defined inside the game. The update function is called 60 times a second from the app. In this update function we pass the current observation to the TD3 algorithm/brain. The network updates with the action to be taken. 
* ai.py

  The brain of the network, this contains the actor, critic, replay buffer and TD3. The convolutional block is added with the Actor and critics. The TD3 is the one which is updating the actions based on the future rewards on taking a perticular step in currentObservation.
  
 ### Current Status
  Implemented the environment and able to move the car in the environment. Implemeted the end to end workflow for TD3 and other modules. Need to play with the parameters, and find better results.
 
 ### Conclusion 
 Learned a lot about TD3 algorithm, making the environment, working on kivy etc. Got enough time to learn about the implementation part and the approch required for making the custom environment and implementing TD3; but was unable to obtain satisfactory results. Will be working again to solve this project.
