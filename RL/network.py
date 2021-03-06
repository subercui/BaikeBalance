# -*- coding: utf-8 -*-
"""
Baiclly OPDAC(Off-Policy deterministic actor-critic) from
"https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf"
Code for Reinforcement Learning:
    according to deep_q_rl 
    
Author: Suber Cui
Start on 2016.03.17
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from updates import opdac_rmsprop,deepmind_rmsprop

class Networks(object):
    """including both actor:u_net and critic:q_net 
    """
    
    def __init__(self, state_width, action_width, action_bound,
                 num_frames, discount, learning_rate, u_lr,rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng):
        self.state_width = state_width
        self.action_width = action_width
        self.action_bound = action_bound#TODO: 没用上
        self.num_frames = num_frames#就是phi_length
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.u_lr = u_lr
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0
        
        ######init u_net######
        """初始化策略网络u_net,包括
        构造state等的符号变量和shared变量，
        构造网络
        给出动作u_acts
        给出网络参数u_params
        """
        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.matrix('actions')
        terminals = T.icol('terminals')
        
        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, state_width, 1),
                     dtype=theano.config.floatX))#是上面这四个维度

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, state_width, 1),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, action_width), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        
        self.u_l_out=self.build_u_network(network_type,state_width,1,
                                         action_width, num_frames, batch_size)
                                         
        u_acts = lasagne.layers.get_output(self.u_l_out, states )
        
        u_params = lasagne.layers.helper.get_all_params(self.u_l_out)
        
        ######------######


        ######init q_net#####
        """初始化评价网络q_net,包括
        构造网络
        给出评价q_vals
        给出下一时刻的评价next_q_vals
        给出td error：diff
        给出网络参数u_params
        有了以上两个经过中间变量q_loss的计算，给出q_updates
        """

        self.q_l_out,in_l1,in_l2=self.build_q_network(network_type, state_width, 1,
                                        action_width, num_frames, batch_size)

        if self.freeze_interval > 0:#这是什么？
            self.next_q_l_out = self.build_q_network(network_type, state_width, 1,
                                        action_width, num_frames, batch_size)
            self.reset_q_hat()
            
        #输入在下面自己定义，注意有state和actions两个都是输入;输出要是（batch*1）的；注意这里action要用输入的真action
        q_vals = lasagne.layers.get_output(self.q_l_out,{in_l1:states,in_l2:actions})#TODO: 现在的问题就是这一句该怎么写
        
        if self.freeze_interval > 0:#这是什么？
            next_q_vals = lasagne.layers.get_output(self.next_q_l_out,{in_l1:next_states,in_l2:u_acts})
        else:
            next_q_vals = lasagne.layers.get_output(self.q_l_out,{in_l1:next_states,in_l2:u_acts})
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        
        #DPG中公式（16）的delta_t,这里和DQN很不同
        diff=(rewards + (T.ones_like(terminals) - terminals)*self.discount *next_q_vals)-q_vals
        
        #17,18两个公式自己写吧，要直接卸T.grad对公式里两个求梯度部分自己求了。另外17式怎么出来的
        
        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            q_loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            q_loss = 0.5 * diff ** 2#果然目标函数q_loss主要就是diff，是sita的函数。反正是求偏导，等于当做reward是与sita无关的量（定量）。

        if batch_accumulator == 'sum':
            q_loss = T.sum(q_loss)#shape (1)
        elif batch_accumulator == 'mean':
            q_loss = T.mean(q_loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))
            
        q_params = lasagne.layers.helper.get_all_params(self.q_l_out)  
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        if update_rule == 'deepmind_rmsprop':
            q_updates = deepmind_rmsprop(q_loss, q_params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            q_updates = lasagne.updates.rmsprop(q_loss, q_params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            q_updates = lasagne.updates.sgd(q_loss, q_params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
        

        ######------######        
        """
        先给出u_updates（由于u_updates和q_upates有依赖，放在这里才给出）
        给出总的updates，于是就能够训练了
        给出符号函数_train
        给出网络输出的符号函数get_u_acts和get_q_vals
        """
        #忽略124-136，重写updates;
        #比如这里q_loss对q_params求导
        #opdac_rmsprop 完成公式(18)
        if batch_accumulator == 'sum':
            acm_u_acts = T.sum(u_acts)#这里先这么粗暴的写了，在acts只有一维的时候可以这样shape (0)
            acm_q=T.sum(q_vals)
        elif batch_accumulator == 'mean':
            acm_u_acts = T.mean(u_acts)
            acm_q=T.mean(q_vals)

        u_updates = opdac_rmsprop(acm_q, actions, acm_u_acts, u_params,self.u_lr,
                                  False)#TODO: 这里该不该填states，还是该填states_shared
        
        self.get_u_acts = theano.function([], u_acts,
                                       givens={states: self.states_shared}) 

        #这个函数get_q_vals或许用不上
        self.get_q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared, actions: self.actions_shared})       
        
        #另一种表达写法updates=OrderedDict(q_updates,**u_updates),意思都是合并两个字典
        updates = OrderedDict(q_updates.items()+u_updates.items())
        
        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        #这个就是公式(16)(17)啦
        self._train = theano.function([], [q_loss, q_vals], updates=updates,
                                      givens=givens)#哦！！！你这样拿givens换就可以每次给进来新的值；
                                                    #可是为什么用givens，为什么不在输入直接写tensorvariable，说是不是你不知道这么写
    

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        #TODO: 这句的功能是在有freeze，即next网络会被冻住的时候，在适当的时候
        #解冻更新；这个逻辑暂时没有用到，之后看一下是否添加
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state,action):
        states = np.zeros((self.batch_size, self.num_frames, self.state_width,
                           1), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        actions = np.zeros((self.batch_size, self.action_width), dtype=theano.config.floatX)
        actions[0, ...] = action
        self.actions_shared.set_value(actions)
        return self.get_q_vals()[0]

    def u_acts(self, state):
        states = np.zeros((self.batch_size, self.num_frames, self.state_width,
                           1), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        return self.get_u_acts()[0]

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            #TODO: 这也不是个很可行的explore策略
            return (self.rng.rand()-0.5)*self.action_bound/10.
        u_act = self.u_acts(state)
        return u_act

    def reset_q_hat(self):
        """force to reset the next_q_l_out network's parameters,
        as the q_l_out network's parameters.
        """
        all_params = lasagne.layers.helper.get_all_param_values(self.q_l_out)
        lasagne.layers.helper.set_all_param_values(self.next_q_l_out, all_params)

                                         
    def build_q_network(self, network_type, state_width, state_height, action_width,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        in_l1 = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, state_width, state_height),
            )
        
        in_l2 = lasagne.layers.InputLayer(
            shape=(batch_size, action_width)
            )

        #如何接受两个输入层，见于lasagne tutorial：All layers work this way, 
        #except for layers that merge multiple inputs: those accept a list of incoming layers as their first constructor argument instead
        rsp_l1 = lasagne.layers.ReshapeLayer(in_l1, ([0],-1))
        l_mg=lasagne.layers.ConcatLayer([rsp_l1,in_l2],axis=1)
        l3 = lasagne.layers.DenseLayer(
            l_mg,
            num_units=1,
            nonlinearity=lasagne.nonlinearities.rectify)
        l_out = lasagne.layers.DenseLayer(
            l3,
            num_units=1,
            nonlinearity=None)

        return l_out,in_l1,in_l2
  
    def build_u_network(self, network_type, state_width, state_height, action_width,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, state_width, state_height)
        )       
        l_h1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)
        l_h2 = lasagne.layers.DenseLayer(
            l_h1,
            num_units=20,
            nonlinearity=lasagne.nonlinearities.rectify)
        l_out = lasagne.layers.DenseLayer(
            l_h2,
            num_units=1,
            nonlinearity=None)

        return l_out

def main():
    net = Networks(state_width=5,action_width=1,action_bound=40,num_frames=1,
                    discount=0.95,learning_rate=.0002,u_lr=.0002,rho=.99,rms_epsilon=1e-6,
                    momentum=0,clip_delta=0, freeze_interval=-1,batch_size=32, 
                    network_type='linear',update_rule='rmsprop',
                    batch_accumulator='mean',rng=np.random.RandomState())
    return net


if __name__ == '__main__':
    net=main()
