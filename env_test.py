import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

env = gym.make('highway-v0', render_mode='rgb_array')

env.configure({"controlled_vehicles": 2})  # Two controlled vehicles
env.configure({"vehicles_count": 1})  # A single other vehicle, for the sake of visualisation
env.configure({"lanes_count": 10})  # A single other vehicle, for the sake of visualisation
env.configure({"duration": 100})  # A single other vehicle, for the sake of visualisation
env.configure({"show_trajectories": True})  # A single other vehicle, for the sake of visualisation


# Multi-agent environment configuration
env.configure({
  "controlled_vehicles": 2,
  "observation": {
    "type": "MultiAgentObservation",
    "observation_config": {
      "type": "Kinematics",
    }
  },
  "action": {
    "type": "MultiAgentAction",
    "action_config": {
    "type": "DiscreteMetaAction",
    "longitudinal": True,
    "lateral": True,
    "target_speeds": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 
                      1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    }
  }
})
env.configure({
    "screen_width": 640,
    "screen_height": 480
})
env.configure({
  "observation": {
    "type": "MultiAgentObservation",
    "observation_config": {
      "type": "Kinematics",
    }
  }
})
obs, info = env.reset(seed=0)

for i in range(100):

    obs, reward, done, truncated, info = env.step((1,1))
    env.render()
    # plt.imshow(obs[i,].T, cmap=plt.get_cmap('gray'))
print(obs)

# plt.imshow(env.render())
# plt.show()