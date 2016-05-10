# -*- coding: utf-8 -*-
"""
Code for Reinforcement Learning:
    according to deep_q_rl 
    
Author: Suber Cui
Start on 2016.05.06
"""
class ALEExperiment(object):
    def __init__(self,agent):
        self.agent=agent
    
    
    def run(self):
        """
        Run the desired number of training epochs, 这是运行时跑的函数
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)