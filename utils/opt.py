"""
SUMMARY:  optimization package
AUTHOR:   Qiuqiang Kong
Created:  2016.03.10 SGD, Momentum, Adagrad
Modified: 2016.03.13 Adadelta, RMSprop
          2016.07.23 G -> Gs
          2016.11.14 add Base class, add reset() to all optimizers
"""
import numpy as np


class Base(object):
    def _reset_memory(self, memory):
        for i1 in range(len(memory)):
            memory[i1] = np.zeros(memory[i1].shape)
        

class SGD():
    def __init__(self, lr):
        self.lr = lr
        
    def GetNewParams(self, params, gparams):
        new_params = []
        for i1 in range(len(params)):
            new_params += [params[i1] - self.lr * gparams[i1]]
        return new_params
        
    
'''
When the target function is very flat, using momentum is better, which can speed up optimization. 
''' 
class Momentum(Base):
    def __init__(self, gamma, lr):
        self.v_ = []
        self.gamma, self.lr = gamma, lr
        
    def GetNewParams(self, params, gparams):
        if not self.v_:
            for param in params:
                self.v_ += [np.zeros_like(param)]
        
        new_params = []
        for i1 in range(len(params)):
            self.v_[i1] = self.gamma * self.v_[i1] + self.lr * gparams[i1]
            new_params += [params[i1] - self.v_[i1]]
        
        return new_params
        
    def reset(self):
        self._reset_memory(self.v_)
        

class Adagrad(Base):
    def __init__(self, lr=0.01):
        self.Gs = []
        self.lr = lr
        self.eps = 1e-6

    def GetNewParams(self, params, gparams):
        if not self.Gs:
            for param in params:
                self.Gs += [np.zeros_like(param)]
                
        new_params = []
        for i1 in range(len(params)):
            self.Gs[i1] += gparams[i1] ** 2
            new_params += [params[i1] - self.lr * gparams[i1] / np.sqrt(self.Gs[i1] + self.eps)]
            
        return new_params
        
    def reset(self):
        self._reset_memory(self.Gs)

        
class Adadelta(Base):
    def __init__(self):
        self.Egs = []
        self.Exs = []
        self.eps = 1e-8
        self.rou = 0.95
        
    def GetNewParams(self, params, gparams):
        if not self.Egs:
            for param in params:
                self.Egs += [np.zeros_like(param)]
                self.Exs += [np.zeros_like(param)]
            
        new_params = []
        for i1 in range(len(params)):
            self.Egs[i1] = self.rou * self.Egs[i1] + (1 - self.rou) * gparams[i1]**2
            delta_x = - (np.sqrt(self.Exs[i1] + self.eps)) / np.sqrt(self.Egs[i1] + self.eps) * gparams[i1]
            self.Exs[i1] = self.rou * self.Exs[i1] + (1 - self.rou) * delta_x**2
            new_params += [params[i1] + delta_x]

        return new_params
        
    def reset(self):
        self._reset_memory(self.Egs)
        self._reset_memory(self.Exs)
            
class RMSprop(Base):
    def __init__(self):
        self.Egs = []
        self.eps = 1e-6
        self.lr = 1e-3
        self.rou = 0.9
        
    def GetNewParams(self, params, gparams):
        if not self.Egs:
            for param in params:
                self.Egs += [np.zeros_like(param)]
                
        new_params = []
        for i1 in range(len(params)):
            self.Egs[i1] = self.rou * self.Egs[i1] + (1 - self.rou) * gparams[i1]**2
            new_params += [params[i1] - self.lr / np.sqrt(self.Egs[i1] + self.eps) * gparams[i1]]
            
        return new_params
        
    def reset(self):
        self._reset_memory(self.Egs)
        
class Adam(Base):
    def __init__(self):
        self.ms = []
        self.vs = []
        self.alpha = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        #self.epoch = 1
        self.iter = 0
        
        
    def GetNewParams(self, params, gparams):
        if not self.ms:
            for param in params:
                self.ms += [np.zeros_like(param)]
                self.vs += [np.zeros_like(param)]
                
        # # origin adam
        # new_params = []
        # self.iter += 1
        # for i1 in range(len(params)):
        #     self.ms[i1] = self.beta1 * self.ms[i1] + (1 - self.beta1) * gparams[i1]
        #     self.vs[i1] = self.beta2 * self.vs[i1] + (1 - self.beta2) * gparams[i1]**2
        #     m_unbias = self.ms[i1] / (1 - np.power(self.beta1, self.iter))
        #     v_unbias = self.vs[i1] / (1 - np.power(self.beta2, self.iter))
        #     new_params += [params[i1] - self.alpha * m_unbias / (np.sqrt(v_unbias) + self.eps)]
            
        
        # fast adam, faster than origin adam
        self.iter += 1
        new_params = []
        alpha_t = self.alpha * np.sqrt(1 - np.power(self.beta2, self.iter)) / (1 - np.power(self.beta1, self.iter))
        for i1 in range(len(params)):
            self.ms[i1] = self.beta1 * self.ms[i1] + (1 - self.beta1) * gparams[i1]
            self.vs[i1] = self.beta2 * self.vs[i1] + (1 - self.beta2) * np.square(gparams[i1])
            new_params += [params[i1] - alpha_t * self.ms[i1] / (np.sqrt(self.vs[i1] + self.eps))]
            
        return new_params
        
    def reset(self):
        self._reset_memory(self.ms)
        self._reset_memory(self.vs)
        self.epoch = 1