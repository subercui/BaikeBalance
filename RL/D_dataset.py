# -*- coding: utf-8 -*-
"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
Created on Tue May 10 17:13:00 2016

@author: Subercui
"""
import numpy as np
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.循环队列

    """
    def __init__(self, state_width, action_width, rng, max_steps=1000, phi_length=1):
        """Construct a DataSet.

        Arguments:
            state_width:vector length
            action_width:vector length
            max_steps - the number of time steps to store
            phi_length - number of states to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
         # Store arguments.
        self.state_width = state_width
        self.action_width = action_width
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.states = np.zeros((max_steps, state_width), dtype=floatX)
        self.actions = np.zeros((max_steps,action_width), dtype=floatX)
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0
        
    def add_sample(self, state, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps
        
    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)
        
    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.states.take(indexes, axis=0, mode='wrap')
        
    def phi(self, state):#你这拼一个任意地方来的img在最后面干什么呢？
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.states.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = state
        return phi
        
    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

        """
        # TODO: 注意这里的维度要和network对应
        # Allocate the response.
        states = np.zeros((batch_size,
                           self.phi_length,
                           self.state_width,
                           1),
                          dtype=floatX)
        actions = np.zeros((batch_size, self.action_width), dtype=floatX)
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')
        next_states = np.zeros((batch_size,
                                self.phi_length,
                                self.state_width,
                                1),
                               dtype=floatX)

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            initial_indices = np.arange(index, index + self.phi_length)
            transition_indices = initial_indices + 1
            end_index = index + self.phi_length - 1
            
            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            states[count] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.imgs.take(transition_indices,
                                                axis=0,
                                                mode='wrap')
            count += 1

        return states, actions, rewards, next_states, terminal