import argparse

import numpy as np
import pandas as pd
from pandas import Series

import json
import random
import torch
import utils
from agent import Agent
from brain import Brain
from consts import *
from env import Environment
from log import Log
from pool import Pool

import testing

class Train():
    def __init__(self, sub, loo):
        self.sub = sub
        self.loo = loo
        self.data_file = './sub{}/pca_loo_tr{}'.format(self.sub, self.loo)
        self.data     = pd.read_pickle(self.data_file)
        self.data_val = pd.read_pickle(self.data_file)
        self.costs = pd.Series(np.ones(32))
        self.pool  = Pool(POOL_SIZE)          # 1.000.000
        self.env   = Environment(self.data, self.costs, FEATURE_FACTOR)
        self.brain = Brain(self.pool)
        self.agent = Agent(self.env, self.pool, self.brain)
        self.log   = Log(self.data_val, self.costs, FEATURE_FACTOR, self.brain)
#    
    def is_time(self, epoch, trigger):
        return (trigger > 0) and (epoch % trigger == 0)

    def run(self):
        epoch_start = 0
        self.brain.update_lr(epoch_start)
        self.agent.update_epsilon(epoch_start)
        
        print("Initializing pool..")
        for i in range(POOL_SIZE // AGENTS):
        	utils.print_progress(i, POOL_SIZE // AGENTS)
        	self.agent.step()
        self.pool.cuda()

        print("Starting..")
        for epoch in range(epoch_start + 1, TRAINING_EPOCHS + 1):
        	# SAVE
        	if self.is_time(epoch, SAVE_EPOCHS):
        		self.brain._save()   #  SAVE_EPOCHS 마다 저장
        
        		save_data = {}
        		save_data['epoch'] = epoch
                
        		with open('run.state', 'w') as file:
        			json.dump(save_data, file)
        
        	# SET VALUES
        	if self.is_time(epoch, EPSILON_UPDATE_EPOCHS):
        		self.agent.update_epsilon(epoch)
        
        	if self.is_time(epoch, LR_SC_EPOCHS):
        		self.brain.update_lr(epoch)
        
        	# LOG
        	if self.is_time(epoch, LOG_EPOCHS):
        		print("Epoch: {}/{}".format(epoch, TRAINING_EPOCHS))
        		self.log.log()
        		self.log.print_speed()
        
        	if self.is_time(epoch, LOG_PERF_EPOCHS):
        		self.log.log_perf()
        
        
        	# TRAIN
        	self.brain.train()
        	
        	for i in range(EPOCH_STEPS):
        		self.agent.step()
                
        test = testing.Test(self.sub, self.loo)
        test.run()
        del test