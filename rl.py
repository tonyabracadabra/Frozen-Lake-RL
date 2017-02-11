# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import deeprl_hw1.lake_envs as lake_env
import numpy as np
import gym
import time

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    
    #initializes value function to 0
    value_func = np.zeros(env.nS,)

    #iterate max_iterations times the Bellman Equation
    for i in range(max_iterations):
      prev_value_func = value_func.copy()
      # marginalize out the inside s' and r to get q (state-action value func), shape = (nS, nA)
      # env.R + gamma * value_func can automatically broadcast the outer dimension
      q = np.einsum("ijk,ijk -> ij", env.T, env.R + gamma * value_func)
      # marginalize out a, shape = (nS,)
      value_func = np.sum(one_hot_encode(env, policy) * q, 1)

      if np.max(np.abs(prev_value_func-value_func)) < tol:
        break

    return value_func

def one_hot_encode(env, policy):
  """One hot encode the policy

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    policy: np.array shape = (nS,)
      The policy to evaluate. Maps states to actions.

    Returns
    -------
    np.ndarray shape = (nS, nA)
      One hot encoded policy
    """
  one_hot_policy = np.zeros((env.nS, env.nA))
  one_hot_policy[np.arange(env.nS), policy] = 1

  return one_hot_policy

def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    

    # shape = (nS, nA)
    q = np.einsum("ijk,ijk -> ij", env.T, env.R + gamma * value_function)
    policy = np.argmax(q, axis=1)

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    new_policy = value_function_to_policy(env, gamma, value_func)

    return (not np.array_equal(policy, new_policy)), new_policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """

    # initialize the policy with random actions
    # policy = np.random.randint(env.nA, size = env.nS)
    # print(policy)
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    for iteration in xrange(max_iterations):
      prev_policy = policy.copy()
      value_func = evaluate_policy(env, gamma, policy)

      changed, policy = improve_policy(env, gamma, value_func, policy)
      print(policy)

      if not changed:
        break
    print(iteration)

    return policy, value_func, 0, 0


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    value_func = np.zeros(env.nS)

    for iteration in xrange(max_iterations):
      # shape = (nS, nA)
      prev_value_func = value_func.copy()
      q = np.einsum("ijk,ijk -> ij", env.T, env.R + gamma * value_func)
      value_func = np.max(q, axis=1)

      if np.max(np.abs(value_func-prev_value_func)) < tol:
        break

    return value_func, iteration

def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)

def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # env = gym.make('Queue-1-v0')
    # uncomment next line to try the deterministic version
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')

    # print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)

    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = env.T.copy()

    for state in xrange(env.nS):
      for action in xrange(env.nA):
        transitions = env.P[state][action]
        for transition in transitions:
          prob, next_state, reward, _ = transition
          env.T[state, action, next_state] = prob
          env.R[state, action, next_state] = reward
        # normalization
        env.T[state, action, :] /= np.sum(env.T[state, action, :])

    gamma = 0.9
    opt_policy, value_func, _, _ = policy_iteration(env, gamma)

    value_func, _ = value_iteration(env, gamma)
    # opt_policy = value_function_to_policy(env, gamma, value_func)
    print(opt_policy)

    env.render()
    s, r, d, _= env.step(opt_policy[env.s])
    while d is False:
      env.render()
      s, r, d, _ = env.step(opt_policy[s])
      time.sleep(1)


    # print('Agent received total reward of: %f' % total_reward)
    # print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
