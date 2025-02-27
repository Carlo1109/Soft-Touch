# Soft Touch

## Project Description
This project was developed for the Medical Robotics exam at Sapienza University of Rome, Master Degree in Control Engineering. The main goal was to interact and to grasp deformable objects with a virtual hand in order to enhance simulations for surgeries, using the Oculus Rift S and  WeArt TouchDIVER G1. 
In "COPPELIA" directory, the hand is fixed in the space and, in front of it, there is a soft cube which can be slightly deformed.
In the virtual environment created in "MUJOCO" directory, it is possible to interact with some simple geometries (e.g. cubes, cylinder, sphere) in order to try the haptic feedback through the glove. Moreover, it is implemented a 3D abdomen's phantom: it is possible to touch the left kidney of the abdomen and experience the feeling of a deformable object. 
As said above, WeArt glove is used to track real hand's closure and to give haptic feedback, from simple force feedback to temperature feedback. Oculus is used to visualize the virtual environment in which objects are rendered and through its controller it is possible to track real hand's position.

## Installation
Once you have downloaded the code in this repositories, please be sure to have the following dependences in Python:
- [WEART Python SDK](https://github.com/WEARTHaptics/WEART-SDK-Python)
    - for the TouchDIVER
    - `pip install weartsdk-sky`
- [CoppeliaSim ZMQ API](https://coppeliarobotics.com/)
    - for the simulation
    - `pip install coppeliasim_zmqremoteapi_client`
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html)
    - for the simulation
    - `pip install mujoco`
- [pyopenxr](https://github.com/cmbruns/pyopenxr/)
    - for the VR headset and the controllers
    - `pip install pyopenxr`
- [pynput](https://pypi.org/project/pynput/)
    - to listen to keyboard press
    - `pip install pynput`
- [matplotlib](https://pypi.org/project/matplotlib/)
    - for real-time performance plots
    - `pip install matplotlib`

Finally, you need the WeArt MiddleWare (only available on Windows) and the Meta Quest Link to use the Oculus.
## Usage
### CoppeliaSim simulation
1. Open the [scene](<CoppeliaSim/Finger_V07_Python.ttt>) in CoppeliaSim.
1. Open the WEART Middleware and connect the TouchDIVER.
1. Launch the [`simulator.py`](simulation.py) file.

### MuJoCo simulation
1. Open the WEART Middleware and connect the TouchDIVERs
1. Connect your VR device and launch the Meta Quest Link.
1. Change the options in [`simulator.py`](simulator.py) according to the scene you want, if you want to use WEART or/and the Oculus and general during simulation set-up.
1. Launch the [`simulator.py`](simulator.py) python file.


## Authors
- b0jck
- Carlo1109
- Milly37
- PiNo010




