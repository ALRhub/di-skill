from fancy_gym import make

# env = make('BoxPushingDenseReplanProDMP-v0', 0)
# env = make('TableTennis4DProDMP-v0', 0)
# env = make('BoxPushingTemporalSparseRotInv-v0', 0)
# env = make('BoxPushingTemporalSparseRnd2Rnd-v0', 0)
# env = make('BoxPushingTemporalSparseNoGuidanceAtAllRotInv-v0', 0)
# env = make('BoxPushingTemporalSparseNotInclinedInit-v0', 0)
# env = make('BoxPushingObstacleTemporalSparse-v0', 0)
# env = make('TableTennis5D-v0', 0)
# env = make('MiniGolf-v2', 0)
# env = make('HopperJumpSparseProMP-v0', 0)
env = make('BeerPongProMP-v0', 0)

env.reset()
for _ in range(10000):
    # env.reset()
    env.render(mode='human')
    obs, r, done, infos = env.step(env.action_space.sample())
    if done:
        env.reset()
print('asdf')
