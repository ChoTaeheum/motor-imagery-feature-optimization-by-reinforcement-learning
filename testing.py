import pandas as pd
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils as utils
import argparse

from openpyxl import load_workbook

class Var():
    def __init__(self):
        self.feature_factor   =   0.00
        self.reward_correct   =   0
        self.reward_incorrect =  -1
        
        self.model_file = "./model"
        
        self.nn_fc_density    = 64
        self.nn_hidden_layers = 2

        self.classes = 2
        self.feature_dim = 32
        self.state_dim = self.feature_dim * 2
        self.action_dim = self.feature_dim + self.classes
        
        self.max_mask_const = 1.e6
        self.chunk_size = 1000
                
        self.costs = pd.Series(np.ones(32))

#==============================
class PerfAgent():
	def __init__(self, env, brain):
		self.env  = env
		self.brain = brain
		self.agents = self.env.agents
		self.done = np.zeros(self.agents, dtype=np.bool)
		self.total_r = np.zeros(self.agents)
		self.total_corr  = np.zeros(self.agents, dtype=np.int32)
		self.total_len  = np.zeros(self.agents, dtype=np.int32)
		self.s = self.env.reset()
        
		self.var = Var()
		self.selected = []

	def act(self, s):
		m = np.zeros((self.agents, self.var.action_dim))	# create max_mask
		m[:, self.var.classes:] = s[:, self.var.feature_dim:]

		p = self.brain.predict(s) - self.var.max_mask_const * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		self.selected.append(a[0])
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done
		self.done = self.done | done
		self.total_len = self.total_len + ~done	    # classification action not counted
		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_corr = self.total_corr + (r == self.var.reward_correct) * newly_finished
        
	def run(self):
		lens = []

		while not np.all(self.done):
			self.step()

		avg_r    = np.sum(self.total_r)
		avg_len  = np.sum(self.total_len)    # 각 데이터마다 feature를 몇개 사용하였나
		avg_corr = np.sum(self.total_corr)
		        
		lens.append(self.total_len)

		return avg_r, avg_len, avg_corr, lens

#==============================
class PerfEnv:
	def __init__(self, data_x, data_y, costs, ff):
		self.x = data_x
		self.y = data_y    # [0]
		self.costs = costs.as_matrix()

		self.agents = len(data_x)    # 33
		self.lin_array = np.arange(self.agents)

		self.ff = ff
		self.var = Var()

	def reset(self):
		self.mask = np.zeros((self.agents, self.var.feature_dim))
		self.done = np.zeros( self.agents, dtype=np.bool )

		return self._get_state()

	def step(self, action):
		self.mask[self.lin_array, action - self.var.classes] = 1

		r = -self.costs[action - self.var.classes] * self.ff

		for i in np.where(action < self.var.classes)[0]:
			r[i] = self.var.reward_correct if action[i] == self.y else self.var.reward_incorrect
			self.done[i] = 1

		s_ = self._get_state()
		return (s_, r, self.done)

	def _get_state(self):
		x_ = self.x * self.mask
		x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32)
		return x_

#==============================
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.var = Var()
		in_nn  = self.var.state_dim
		out_nn = 64

		self.l_fc = []
		for i in range(self.var.nn_hidden_layers):
			l = torch.nn.Linear(in_nn, out_nn)
			in_nn = out_nn

			self.l_fc.append(l)
			self.add_module("l_fc_"+str(i), l)

		self.l_out_q_val = torch.nn.Linear(in_nn, self.var.action_dim)		# q-value prediction

	def forward(self, batch):
		flow = batch

		for l in self.l_fc:
			flow = F.relu(l(flow))

		a_out_q_val = self.l_out_q_val(flow)

		return a_out_q_val

#==============================
class Brain:
	def __init__(self):
		self.model  = Net()
		print("Network architecture:\n"+str(self.model))
		self.var = Var()

	def _load(self):
		self.model.load_state_dict( torch.load(self.var.model_file, map_location={'cuda:0': 'cpu'}) )

	def predict(self, s):
		s = Variable(torch.from_numpy(s))
		res = self.model(s)
		return res.data.numpy()


class Test():
    def __init__(self, sub, loo):
        self.sub = sub
        self.loo = loo
        self.data_file  = "./sub{}/pca_loo_ev{}".format(self.sub, self.loo)
        self.data = pd.read_pickle(self.data_file)
        
        self.var = Var()
        self.data_x = self.data.iloc[:, 0:-1].astype('float32').as_matrix()
        self.data_y = self.data.iloc[:,   -1].astype('int32').as_matrix()
        
        self.data_len = len(self.data)
        #=============================        
        self.brain = Brain()
        self.brain._load()        
        self.confidence = np.zeros(self.data_len)
        self.avg_r = 0
        self.avg_len = 0
        self.avg_corr = 0
        self.idx = 0
        self.lens = []
        self.selected_features = []
    
    def run(self):        
        while self.idx < self.data_len:
        	utils.print_progress(self.idx, self.data_len, step=self.var.chunk_size)
            
        	self.last = min(self.idx + self.var.chunk_size, self.data_len)
        	self.env = PerfEnv(self.data_x, self.data_y, self.var.costs, self.var.feature_factor)
        	self.agent = PerfAgent(self.env, self.brain)
        
        	_r, _len, _corr, _lens = self.agent.run()
        	self.avg_r += _r
        	self.avg_len += _len
        	self.avg_corr += _corr
        	self.lens.append(_lens)
        	self.idx += self.var.chunk_size
        
        self.avg_len  /= self.data_len
        self.avg_r    /= self.data_len
        self.avg_corr /= self.data_len
        
        print('====================================')
        print("R: ", self.avg_r)
        print("Correct: ", self.avg_corr)
        print("Selected Features: ", [a-1 for a in self.agent.selected[:-1]])
        print("Length: ", self.avg_len)
        print("FEATURE_FACTOR: ", self.var.feature_factor)
        print("loo: ", self.loo)
        print('====================================')
        
        rawresult = pd.DataFrame(np.reshape(np.array([self.loo, self.var.feature_factor, self.avg_corr, self.avg_len]), (1, 4)))
        selected = np.array([a-1 for a in self.agent.selected[:-1]])
        
        e = open('rawresult.csv', 'w')
        e.write(str(rawresult))
        e.close()
        
        f = open('selected.csv', 'w')
        f.write(str(selected))
        f.close()
                
        self.lens = np.concatenate(self.lens, axis=1).flatten()
        np.set_printoptions(suppress=True, threshold=1e8, precision=4)
        
        with open('histogram', 'w') as file:
        	for x in self.lens:
        		file.write("{} ".format(x))

