# -*- coding: utf-8 -*-
"""
The Experiment class handles the logic for training a OPDAC in the environment

Code for Reinforcement Learning:
    according to deep_q_rl 
    
Author: Suber Cui
Start on 2016.05.06
"""
class Experiment(object):
    def __init__(self,agent,num_epochs,terminal_thres):
        self.serial=serial
        self.agent=agent
        self.num_epochs=num_epochs
        self.terminal_thres=terminal_thres
    
    def run(self):
        """
        Run the desired number of training epochs, 这是运行时跑的函数
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)
#            先不说test的事情，下面四行是每个epoch之后进行test
#            if self.test_length > 0:
#                self.agent.start_testing()
#                self.run_epoch(epoch, self.test_length, True)
#                self.agent.finish_testing(epoch)
            
    def run_epoch(self,epoch,num_steps,testing=False):
        #几个比如经过terminal的episode组成epoch
        steps_left=num_steps
        while steps_left>0:
            prefix = "testing" if testing else "training"
            logging.info(prefix+"epoch: "+str(epoch)+"steps_left in this epoch: "+
                         str(steps_left))
            _, num_steps=self.run_episode(steps_left,testing)
            
            steps_left-=num_steps
            
        
    def _step(self, action):
        """apply and sendout the action to the environment
        """
        self.serial.send(action)
        
    def _ifterminal(self,reward):
        if reward <self.terminal_thres:
            return True
        else:
            return False
        
        
    def run_episode(self,max_steps,testing):
        """ run a single training episode, till the terminal or the steps end
        """
        reward,observation=self.get_reward_observe()
        action=self.agent.start_episode(observation)
        num_steps=0
        while True:
            self._step(action)
            terminal=self._ifterminal(reward)
            num_steps+=1
            
            if terminal or num_steps >=max_steps:
                self.agent.end_episode(reward,terminal)
                break
            reward,observation=self.get_reward_observe()
            action=self.agent.step(reward, observation)
        return terminal, num_steps
    
    
    def get_reward_observe(self):
        sensory=self.serial.read()
        reward,observe=parsereceive(sensory)
        return reward,observe
        