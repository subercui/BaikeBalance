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
    
def opdac_rmsprop(q_vals, acts_t, u_acts, u_params,learning_rate,WhetherDirect):
    if WhetherDirect:
        #TODO: 这个一定要看一看，原文为什么不能直接求导？,因为u_params根本不在计算过程内
        return sgd(q_vals, u_params, learning_rate)
    else:
        q2a_grads=get_or_compute_grads(q_vals, acts_t)
        a2w_grads=get_or_compute_grads(u_acts, u_params)
        #TODO: 这还有个正负号错误，要增长
        grads=a2w_grads*q2a_grads#TODO: 这两个dict怎么乘！
        
        updates = OrderedDict()
        for param, grad in zip(u_params, grads):
            updates[param] = param - learning_rate * grad

        return updates
        

        
                
        
    
            
def deepmind_rmsprop(loss_or_grads, params, learning_rate, 
                     rho, epsilon):
    """RMSProp updates [1]_.

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients.

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression

    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.

    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:

    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        acc_grad = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        acc_grad_new = rho * acc_grad + (1 - rho) * grad

        acc_rms = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2


        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - learning_rate * 
                          (grad / 
                           T.sqrt(acc_rms_new - acc_grad_new **2 + epsilon)))

    return updates
