# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
from gym.utils import seeding, colorize
import numpy as np


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """

    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def categorical_sample(prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)

        return (csprob_n > np_random.rand()).argmax()

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        self.nS = 648
        self.nA = 4

        self.P = dict()

        self.p1 = p1 * (1 - p2) * (1- p3)
        self.p2 = (1 - p1) * p2 * (1- p3)
        self.p3 = (1 - p1) * (1 - p2) * p3
        self.p_12 = p1 * p2 * (1- p3)
        self.p_13 = p1 * (1 - p2) * p3
        self.p_23 = (1 - p1) * p2 * p3
        self.p_123 = p1 * p2 * p3
        self.p_n = (1 - p1) * (1 - p2) * (1 - p3)

        self.helper = {self.p1:[0,1,0,0], self.p2:[0,0,1,0], self.p3:[0,0,0,1], \
                       self.p_12:[0,1,1,0], self.p_13:[0,1,0,1], self.p_23:[0,0,1,1], \
                       self.p_123:[0,1,1,1], self.p_n:[0,0,0,0]}


    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """

        self.state[0] = 1

        return self.state

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        
        transitions = query_model(self.state, action)

        i = categorical_sample([t[0] for t in transitions], self.np_random)

        prob, next_state, reward, is_terminal = transitions[i]
        self.state = state
        self.last_action = action

        return (nextstate, reward, is_terminal, {"prob" : prob})


    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        q, s1, s2, s3 = self.state

        color = ["white" for i in xrange(3)]
        color[q-1] = "yellow"

        for i in xrange(5,0,-1):
            temp = colorize(u"\u2588 ", color[0]) if s1 >= i else "  "
            temp = temp + colorize(u"\u2588 ", color[1]) if s2 >= i else temp + "  "
            temp = temp + colorize(u"\u2588 ", color[2]) if s3 >= i else temp + "  "
            outfile.write(temp)

        outfile.write(get_action_name(self.last_action))

        return outfile

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """ 

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        
        reward = 0
        if action == SERVICE_QUEUE:
            if state[state[0]] > 1:
                state[state[0]] -= 1
                reward = 1
        else:
            state[0] = action + 1

        transitions = []
        for p, update in self.helper.iteritems():
            new_state = np.array(state) + update
            new_state = [s if s <= 5 else 5 for s in new_state]
            transitions.append((p, new_state, reward, False))

        return transitions

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
