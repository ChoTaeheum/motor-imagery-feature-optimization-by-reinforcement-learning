import numpy as np
import torch

from consts import *

class Pool():
	def __init__(self, size):
		self.data_s  = torch.FloatTensor(size, STATE_DIM)     # (1.000.000, 108)
		self.data_a  = torch.LongTensor(size, 1)              # (1.000.000, 1)
		self.data_r  = torch.FloatTensor(size, 1)             # (1.000.000, 1)
		self.data_s_ = torch.FloatTensor(size, STATE_DIM)     # (1.000.000, 108)

		self.idx  = 0
		self.size = size

	def put(self, x):
		s, a, r, s_ = x
		size = len(s)    # 1000

		self.data_s [self.idx:self.idx+size, :] = torch.from_numpy(s)     # 0 ~ 1000에 (1000, 108)이 들어감
		self.data_a [self.idx:self.idx+size, 0] = torch.from_numpy(a)     # 버전 바뀌면서 1차원 매트릭스의 경우 차원을 정확히 표시해줘야 함
		self.data_r [self.idx:self.idx+size, 0] = torch.from_numpy(r)
		self.data_s_[self.idx:self.idx+size, :] = torch.from_numpy(s_)

		self.idx = (self.idx + size) % self.size

	def sample(self, size):
		# print(self.size, 'self')
		# print(size,'size')
		# idx = torch.from_numpy(np.random.choice(self.size, size)).cuda()     # 전체 경험 리플레이 메모리에서 배치사이즈만큼 랜덤하게 선택
		idx = np.random.choice(self.size, size)     # 전체 경험 리플레이 메모리에서 배치사이즈만큼 랜덤하게 선택
		# print(idx[5])
		return self.data_s[idx], self.data_a[idx], self.data_r[idx], self.data_s_[idx]

	def cuda(self):
		self.data_s  = self.data_s.cuda() 
		self.data_a  = self.data_a.cuda() 
		self.data_r  = self.data_r.cuda() 
		self.data_s_ = self.data_s_.cuda()