#!/usr/bin/env python
# coding: utf-8

# In[20]:
from matplotlib import pyplot as plt
from collections import deque
import numpy as np

import gymnasium as gym
import highway_env


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
     'reward_speed_range': [20, 80],
    #  'other_vehicles_type': 'highway_env.vehicle.behavior.LinearVehicle',
    'controlled_vehicles': 3,
    }
env = gym.make('highway-v0', render_mode='rgb_array')



# # 比例控制

# In[21]:

env.configure(config)
env.reset()

env.road.vehicles[0].speed=0
env.road.vehicles[0].position=[0,12]
env.road.vehicles[1].speed=10
env.road.vehicles[1].position=[20,8]
env.road.vehicles[2].speed=10
env.road.vehicles[2].position=[20,12]

ego_x_his1=[0]
ego_v_his1=[0]
prcd_x_his1=[20]
prcd_y_his1=[]
prcd_v_his1=[10]

his_u1=[]

u=0
dt=1/5
for i in range(1000):   

    action = [u,0]
    obs, reward, done, truncated, info = env.step(action)
    
    ego_vehicle=env.road.vehicles[0]
    prcd_vehicle=env.road.vehicles[1]
    
    ego_v=ego_vehicle.speed
    ego_x=ego_vehicle.position[0]
    
    prcd_v=prcd_vehicle.speed
    prcd_x=prcd_vehicle.position[0]
    
    ego_v_his1.append(ego_v)
    ego_x_his1.append(ego_x)
    prcd_v_his1.append(prcd_v)
    prcd_x_his1.append(prcd_x) 
    
    dx=prcd_x-ego_x
    dv=prcd_v-ego_v
    
    u=1/10*(dx-50+dv)
    
    his_u1.append(u)
    env.render()
    if info['crashed']==True:
        break
env.close()


# # PID

# In[23]:


env.configure(config)
env.reset()

env.road.vehicles[0].speed=0
env.road.vehicles[0].position=[0,12]
env.road.vehicles[1].speed=10
env.road.vehicles[1].position=[20,8]
env.road.vehicles[2].speed=10
env.road.vehicles[2].position=[20,12]



ego_x_his2=[0]
ego_v_his2=[0]
prcd_x_his2=[20]
prcd_y_his2=[]
prcd_v_his2=[10]

his_u2=[]
buffer1 = deque(maxlen=10)
buffer2 = deque(maxlen=10)

u=0
dt=1/5
for _ in range(1000):   
    
    action = [u,0]
    obs, reward, done, truncated, info = env.step(action)
    
    ego_vehicle=env.road.vehicles[0]
    prcd_vehicle=env.road.vehicles[1]
    
    ego_v=ego_vehicle.speed
    ego_x=ego_vehicle.position[0]
    
    prcd_v=prcd_vehicle.speed
    prcd_x=prcd_vehicle.position[0]
    
    ego_v_his2.append(ego_v)
    ego_x_his2.append(ego_x)
    prcd_v_his2.append(prcd_v)
    prcd_x_his2.append(prcd_x) 
    
    ep_x=prcd_x-ego_x-20
    buffer1.append(ep_x)
    ei_x=np.sum(buffer1)*dt  
    if len(buffer1)>=2:        
        ed_x=(buffer1[-1]-buffer1[-2])/dt
    else:
        ed_x=0     
    e_x=(1*ep_x+0.2*ei_x+0.1*ed_x)
    
    ep_v=prcd_v-ego_v
    
    buffer2.append(ep_v)
    ei_v=np.sum(buffer2)*dt    
    if len(buffer2)>=2:        
        ed_v=(buffer2[-1]-buffer2[-2])/dt
    else:
        ed_v=0        
    e_v=(1*ep_v+0.2*ei_v+0.1*ed_v)
    
    
    u=max(min(e_x+e_v,5),-5)
    
    his_u2.append(u)
    env.render()
    if info['crashed']==True:
        break
env.close()


# # MPC

# In[24]:


import casadi as ca


def mpc_controller(xr,dv,dx,N,dt):
    u = ca.MX.sym('u',N) #控制量，未来N个时刻
    v_= dv
    x_= dx #当前时刻的距离差

    obj= 0  #优化函数
    w1=0.7
    w2=1-w1
    weight=[w1,w2]

    #预测
    for step in range(0,N,1):        
        v_=v_-u[step]*dt
        x_=x_+v_*dt  #模型       
        obj=obj+weight[0]*(x_-xr)**2+weight[1]*u[step]**2

    nlp = {'x': u, 'f': obj}

    solver = ca.nlpsol('solver','ipopt', nlp)    
    solution = solver(x0=0,lbx=-5,ubx=5)
    u=solution['x'].full()[0][0]
    return u


# In[26]:


env.configure(config)
env.reset()

env.road.vehicles[0].speed=0
env.road.vehicles[0].position=[0,12]
env.road.vehicles[1].speed=10
env.road.vehicles[1].position=[20,8]
env.road.vehicles[2].speed=10
env.road.vehicles[2].position=[20,12]



ego_x_his3=[0]
ego_v_his3=[0]
prcd_x_his3=[20]
prcd_y_his3=[]
prcd_v_his3=[10]

his_u3=[]

u=0

for _ in range(1000):
    action = [u,0]
    obs, reward, done, truncated, info = env.step(action)
    
    ego_vehicle=env.road.vehicles[0]
    prcd_vehicle=env.road.vehicles[1]
    
    ego_v=ego_vehicle.speed
    ego_x=ego_vehicle.position[0]
    
    prcd_v=prcd_vehicle.speed
    prcd_x=prcd_vehicle.position[0]
    
    ego_v_his3.append(ego_v)
    ego_x_his3.append(ego_x)
    prcd_v_his3.append(prcd_v)
    prcd_x_his3.append(prcd_x)
    
    dx=prcd_x-ego_x
    dv=prcd_v-ego_v
        
    u=mpc_controller(50,dv,dx,20,dt)
    
    his_u3.append(u)
    env.render()
    if info['crashed']==True:
        break
env.close()

# In[28]:


plt.figure(figsize=(8,6))
plt.title('车速对比')
#plt.plot(np.transpose(prcd_v_his2))
plt.plot(np.transpose(ego_v_his1))
plt.plot(np.transpose(ego_v_his2))
plt.plot(np.transpose(ego_v_his3))
plt.xlabel('时间 - s')
plt.ylabel('速度 - m/s')
plt.legend(['K','PID','MPC'])


# In[29]:


plt.figure(figsize=(8,6))
plt.title('与前车距离对比')
plt.plot(np.transpose(prcd_x_his1)-np.transpose(ego_x_his1))
plt.plot(np.transpose(prcd_x_his2)-np.transpose(ego_x_his2))
plt.plot(np.transpose(prcd_x_his3)-np.transpose(ego_x_his3))
plt.ylim([0,40])
plt.xlabel('时间 - s')
plt.ylabel('与前车距离 - m')
plt.legend(['K','PID','MPC'])


# In[30]:


plt.figure(figsize=(8,6))
plt.title('控制量（加速度）对比')
plt.plot(np.transpose(his_u1))
plt.plot(np.transpose(his_u2))
plt.plot(np.transpose(his_u3))
plt.xlabel('时间 - s')
plt.ylabel('加速度 - m/s^2')
plt.legend(['比例控制','PID','MPC'])

# In[ ]:
plt.show()
