# P2_S10

## Autonomous Car using TD3 Reinforcement Learning

Car needs to travel from location A to location B in the city using the TD3 algorithm without using sensors. Using a basic CNN to get sensory data from the map.

Here the current state is represented by a 60x60 image patch on sand image. The patch may contain sand, road with the car. Using this informaton from the environment, we take an action to reach the next state with a reward. The action replay buffer contains these parameters (currentState, action, nextState, reward, done), where done represents the end of an episode. The action predicted by the model is the direction (angle of rotation) and velocity for the vehicle. The done parameter is set when the agent reaches the edge of the image and when it reaches the destination. 
After the car reaches the done state, it is respawn in random location in the map.

ToDo:
Not yet reached proper results
