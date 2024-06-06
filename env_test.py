#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from collections import deque
import numpy as np

import gymnasium as gym
import highway_env 
import casadi as ca
import time



config =     {
    "observation": 
         {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["presence", 'x','y',"vx", "vy"],
        "features_range": 
            {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-100, 100],
            "vy": [-100, 100]
            },
        "absolute": True,
        'normalize':False,
        "order": "sorted"
        },
    "action": {
        "type": "ContinuousAction"#'DiscreteMetaAction' 
    },    
    "lanes_count": 10,
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 5,  # [Hz]
     'vehicles_count':  0,
     'reward_speed_range': [0, 20],
    'controlled_vehicles': 3,
    }
env = gym.make('highway-v0')
env.configure(config)
env.reset()

ego_car = env.road.vehicles[0]
ego_car.speed=0
ego_car.position=[5,12]

tractor = env.road.vehicles[1]
tractor.color = (200,50,20)
tractor.speed=10
tractor.position=[20,8]

blue_car = env.road.vehicles[2]
blue_car.speed=10
blue_car.position=[20,12]
blue_car.color = (0,128,255)





def mpc_controller(yr, xr, dv, dy, dx, N, dt):
    v_ = dv
    x_ = dx  

    obj = 0  
    w1 = 0.5
    w2 = 1 - w1
    weight = [w1, w2]

    # Create optimization problem
    opti = ca.Opti()
    u = opti.variable(N)  # Control inputs
    y_pred = opti.variable(N)  # Predicted outputs

    # Initialize the first predicted output
    opti.subject_to(y_pred[0] == dy)

    for step in range(1, N):
        # Update the predicted velocity and distance for the next step
        v_ = v_ - u[step - 1] * dt
        x_ = x_ + v_ * dt
        y_pred[step] = y_pred[step - 1] + v_ * dt
        # Update the objective function
        obj += weight[0] * ((y_pred[step] - yr) ** 2) + weight[1] * (u[step - 1] ** 2)
        # Add a penalty if y_pred is less than yr
        
    opti.minimize(obj)
    # Constraint to ensure y_pred >= yr
    # Solver settings
    opti.solver('ipopt')
    try:
        sol = opti.solve()
        u_opt = sol.value(u)
        print(u_opt[0])
        return u_opt[0]
    except Exception as e:
        print("Solver failed:", e)
        return None





ego_x_his3=[0]
ego_v_his3=[0]
prcd_x_his3=[20]
prcd_y_his3=[]
prcd_v_his3=[10]
u=0
dt=1/2
his_u3=[]

for i in range(200):
    if i==0:
        env.road.vehicles[2].speed=12

    action = [u,0]
    obs, reward, done, truncated, info = env.step(action)
    
    ego_v=ego_car.speed
    tractor_v=tractor.speed
    ego_x=ego_car.position[0]
    ego_y=ego_car.position[1]

    prcd_v=blue_car.speed
    prcd_x=blue_car.position[0]
    prcd_y=tractor.position[0]

    ego_v_his3.append(ego_v)
    ego_x_his3.append(ego_x)
    prcd_v_his3.append(prcd_v)
    prcd_x_his3.append(prcd_x)
    
    dx=prcd_x-ego_x
    dy=prcd_y-ego_x

    dv=tractor_v-ego_v
        
    u=mpc_controller(0,50,dv,dy,dx,10,dt)

    his_u3.append(u)
    env.render()
    if info['crashed']==True:
        break
env.close()
print(obs)


# Create a figure with 3 subplots arranged in 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Speed plot
axs[0].set_title('Speed')
axs[0].plot(np.transpose(ego_v_his3))  # .T is shorthand for np.transpose
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Speed (m/s)')
axs[0].legend(['MPC'])

# Distance plot
axs[1].set_title('Distance')
axs[1].plot(np.transpose(prcd_x_his3) - np.transpose(ego_x_his3))  # .T is shorthand for np.transpose
axs[1].set_ylim([0, 40])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Distance (m)')
axs[1].legend(['MPC'])

# Acceleration plot
axs[2].set_title('Acceleration')
axs[2].plot(np.transpose(his_u3))  # .T is shorthand for np.transpose
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].legend(['MPC'])

# Adjust layout
plt.tight_layout()

# Show the combined figure
plt.show()

ego_car.speed = 5
print(ego_car.step(0))
print(ego_car.position)
print(ego_car.step(10))
print(ego_car.position)