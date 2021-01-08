import pandas as pd
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils as utils
import argparse

from openpyxl import load_workbook 


FEATURE_FACTOR   =   0.07
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1

DATA_FILE = "./sub1/csp_loo_ev6"
MODEL_FILE = "./model"

META_COSTS = 'cost'
META_AVG   = 'avg'
META_STD   = 'std'

NN_FC_DENSITY = 64
NN_HIDDEN_LAYERS = 2


CLASSES = 2
FEATURE_DIM = 32
STATE_DIM = FEATURE_DIM * 2
ACTION_DIM = FEATURE_DIM + CLASSES

MAX_MASK_CONST = 1.e6
CHUNK_SIZE = 1000

SELECTED_FEATURES = []

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

	def act(self, s):
		m = np.zeros((self.agents, ACTION_DIM))	# create max_mask
		m[:, CLASSES:] = s[:, FEATURE_DIM:]

		p = self.brain.predict(s) - MAX_MASK_CONST * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		SELECTED_FEATURES.append(a[0])
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done
		self.done = self.done | done
		self.total_len = self.total_len + ~done	    # classification action not counted
		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_corr = self.total_corr + (r == REWARD_CORRECT) * newly_finished
        
#		print(self.total_len)

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

	def reset(self):
		self.mask = np.zeros( (self.agents, FEATURE_DIM) )
		self.done = np.zeros( self.agents, dtype=np.bool )

		return self._get_state()

	def step(self, action):
		self.mask[self.lin_array, action - CLASSES] = 1

		r = -self.costs[action - CLASSES] * self.ff

		for i in np.where(action < CLASSES)[0]:
			r[i] = REWARD_CORRECT if action[i] == self.y else REWARD_INCORRECT
#			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
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

		in_nn  = STATE_DIM
		out_nn = NN_FC_DENSITY

		self.l_fc = []
		for i in range(NN_HIDDEN_LAYERS):
			l = torch.nn.Linear(in_nn, out_nn)
			in_nn = out_nn

			self.l_fc.append(l)
			self.add_module("l_fc_"+str(i), l)

		self.l_out_q_val = torch.nn.Linear(in_nn, ACTION_DIM)		# q-value prediction
		# self.cuda()

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

	def _load(self):
		self.model.load_state_dict( torch.load(MODEL_FILE, map_location={'cuda:0': 'cpu'}) )

	def predict(self, s):
		# s = Variable(torch.from_numpy(s).cuda())
		s = Variable(torch.from_numpy(s))
		res = self.model(s)
		# return res.data.cpu().numpy()
		return res.data.numpy()

#==============================
parser = argparse.ArgumentParser()
parser.add_argument('-ff', type=float, help="feature factor")
parser.add_argument('-m', type=str, help="model file")

args = parser.parse_args()

#if args.ff:
#	FEATURE_FACTOR = args.ff
#	print("FEATURE_FACTOR = {}".format(FEATURE_FACTOR))
#
#if args.m:
#	MODEL_FILE = args.m
#	print("MODEL_FILE = {}".format(MODEL_FILE))

#==============================
#if loo == 1:
#	data = pd.read_pickle(DATA_FILE).T    # 여기가 자꾸 모양이 바뀜! 왜 그런걸까?    1번은 (33, 1) <-> 2번부터 (1, 33)
#	costs = pd.Series(np.ones(32))
#
#else:
#	data = pd.read_pickle(DATA_FILE)
#	costs = pd.Series(np.ones(32))
    
data = pd.read_pickle(DATA_FILE)
costs = pd.Series(np.ones(32))


# 2번 부터는 여기가 맞음
data_x   = data.iloc[:, 0:-1].astype('float32').as_matrix()
data_y   = data.iloc[:,   -1].astype('int32').as_matrix()

data_len = len(data)

#==============================
brain = Brain()
brain._load()

confidence = np.zeros(data_len)
avg_r = 0
avg_len = 0
avg_corr = 0
idx = 0
lens = []

while idx < data_len:
	utils.print_progress(idx, data_len, step=CHUNK_SIZE)

	last = min(idx + CHUNK_SIZE, data_len)

	# data_batch_x = data_x[idx:last]
	# data_batch_y = data_y[idx:last]

	data_batch_x = data_x
	data_batch_y = data_y
    
	env = PerfEnv(data_batch_x, data_batch_y, costs, FEATURE_FACTOR)
	agent = PerfAgent(env, brain)

	_r, _len, _corr, _lens = agent.run()
	avg_r += _r
	avg_len += _len
	avg_corr += _corr
	lens.append(_lens)

	idx += CHUNK_SIZE

avg_len  /= data_len
avg_r    /= data_len
avg_corr /= data_len


print('====================================')
print("R: ", avg_r)
print("Correct: ", avg_corr)
print("Selected Features: ", SELECTED_FEATURES[:-1])
print("Length: ", avg_len)
print("FEATURE_FACTOR: ", FEATURE_FACTOR)
print('====================================')

rawresult = pd.DataFrame(np.reshape(np.array([FEATURE_FACTOR, avg_corr, avg_len]), (1, 3)))

e = open('rawresult.txt', 'w')
e.write(str(rawresult))
e.close()

lens = np.concatenate(lens, axis=1).flatten()

np.set_printoptions(suppress=True, threshold=1e8, precision=4)

with open('histogram', 'w') as file:
	for x in lens:
		file.write("{} ".format(x))
	# print(lens, file=file)	


del data, data_x, data_y, data_batch_x, data_batch_y, env, agent
del MODEL_FILE
