# -*- coding: utf-8 -*-
"""
Gradient update rules for the OPDAC. 

Some code here is modified from the Lasagne package:
 
https://github.com/Lasagne/Lasagne/blob/master/LICENSE

Created on Fri May 20 15:35:39 2016

@author: Subercui
"""
import theano
import theano.tensor as T
from lasagne.updates import get_or_compute_grads
from collections import OrderedDict
import numpy as np

#使用这个函数！：get_or_compute_grads，输入要求到的目标和对象，直接给出导数。
#甚至也可以输入导数值“list形式”，直接输出导数值
def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    #现在的目标很简单，就是拿到这个正确的grads    
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates
    
def opdac_rmsprop(q_vals, u_acts, u_params,learning_rate,WhetherDirect):
    if WhetherDirect:
        #TODO: 这个一定要看一看，原文为什么不能直接求导？
        return sgd(q_vals, u_params, learning_rate)
    else:
        q2a_grads=get_or_compute_grads(q_vals, u_acts)
        a2w_grads=get_or_compute_grads(u_acts, u_params)
        grads=a2w_grads*q2a_grads#这两个dict怎么乘！
        
        updates = OrderedDict()
        for param, grad in zip(u_params, grads):
            updates[param] = param - learning_rate * grad

        return updates
       