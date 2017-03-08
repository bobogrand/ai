import gym
import math
env = gym.make('CartPole-v0')
for episode in range(0,100):
        env.reset()
        for _ in range(1000):
                env.render()
                obs, reward,done, _ = env.step(env.action_space.sample()) # take a random action
                if math.fabs(obs[0]) > 3 :
                        done = 0
                        break
