# -*- coding: utf-8 -*-
"""
The wrap of an environment interface

Created on Sun June 5 2016

@author: Subercui

"""
import gym

class EnvInterface(object):
    def __init__(self,usegym=True):
        if usegym:
            self.env=gym.make('CartPole-v0')
            self.action_width=1
            self.observation=self.env.reset()
            self.reward=0
            self.terminated=False
        
    def reset(self):
        self.observation=self.env.reset()
        
    def send(self,action):
        self.env.render()
        #now action is a float
        action=int(action>0)#convert to int
        self.observation, self.reward, self.terminated, info = self.env.step(action)
        if self.terminated:
            print "terminated"
            self.env.reset()
        
        
    def read(self):
        return self.reward,self.observation.reshape((1,-1,1))