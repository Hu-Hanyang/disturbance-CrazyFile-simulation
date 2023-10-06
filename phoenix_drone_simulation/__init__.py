from gym.envs.registration import register


# ==================
#   Hover Task
# ==================

register(
    id='DroneHoverSimpleEnv-v0',
    entry_point='phoenix_drone_simulation.envs.hover:DroneHoverSimpleEnv',
    max_episode_steps=500,
)

register(
    id='DroneHoverBulletEnv-v0',
    entry_point='phoenix_drone_simulation.envs.hover:DroneHoverBulletEnv',
    max_episode_steps=500,
)
#  Hanyang: register our envs here
register(
    id = 'DroneHoverBulletEnvWithAdversary-v0',
    entry_point = 'phoenix_drone_simulation.envs.hover:DroneHoverBulletEnvWithAdversary',
    max_episode_steps = 500,
)

register(
    id='DroneHoverBulletEnvWithRandomHJAdversary-v0',
    entry_point='phoenix_drone_simulation.envs.hover:DroneHoverBulletEnvWithRandomHJAdversary',
    max_episode_steps=500,
)

register(
    id = 'DroneHoverBulletEnvWithoutAdversary-v0',
    entry_point = 'phoenix_drone_simulation.envs.hover:DroneHoverBulletEnvWithoutAdversary',
    max_episode_steps = 500,
)

register(
    id = 'DroneHoverBulletEnvWithRandomAdversary-v0',
    entry_point = 'phoenix_drone_simulation.envs.hover:DroneHoverBulletEnvWithRandomAdversary',
    max_episode_steps = 500,
)

register(
    id='DroneHoverBulletEnvWithAdversaryInitial-v0',
    entry_point = 'phoenix_drone_simulation.envs.hover:DroneHoverBulletEnvWithAdversaryInitial',
    max_episode_steps = 500,
)
# ==================
#   Take-off Task
# ==================

register(
    id='DroneTakeOffSimpleEnv-v0',
    entry_point='phoenix_drone_simulation.envs.takeoff:DroneTakeOffSimpleEnv',
    max_episode_steps=500,
)

register(
    id='DroneTakeOffBulletEnv-v0',
    entry_point='phoenix_drone_simulation.envs.takeoff:DroneTakeOffBulletEnv',
    max_episode_steps=500,
)

# ==================
#   Circle Task
# ==================

register(
    id='DroneCircleSimpleEnv-v0',
    entry_point='phoenix_drone_simulation.envs.circle:DroneCircleSimpleEnv',
    max_episode_steps=500,
)

register(
    id='DroneCircleBulletEnv-v0',
    entry_point='phoenix_drone_simulation.envs.circle:DroneCircleBulletEnv',
    max_episode_steps=500,
)