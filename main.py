import gym
import deeprl_hw1.queue_envs as queue_env

def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    env = gym.make('Queue-1-v0')

    env.render()
    env.step(1)
    env.render()
    env.step(1)
    env.render()
    env.step(2)
    env.render()
    env.step(3)
    env.render()
    env.step(2)

    # total_reward, num_steps = run_random_policy(env)
    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()