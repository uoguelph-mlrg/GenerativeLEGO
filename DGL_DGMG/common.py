import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import copy


class MLP(nn.Module):
	def __init__(self, num_layers, hidden_size, input_size, output_size, batch_norm = False):
		super().__init__()
		self.linears = nn.ModuleList()
		self.batch_norm = batch_norm
		self.relu = nn.LeakyReLU()

		if batch_norm:
			self.bn = nn.ModuleList()
			self.get_bn_func = lambda size: nn.BatchNorm1d(size)
		else:
			self.bn = []
			do_nothing = lambda x: x
			self.get_bn_func = lambda size: do_nothing

		self.num_layers = num_layers
		if num_layers == 1:
			self.linears.append(nn.Linear(input_size, output_size))
			return

		self.linears.append(nn.Linear(input_size, hidden_size))
		self.bn.append(self.get_bn_func(hidden_size))
		for layer in range(num_layers - 2):
			self.linears.append(nn.Linear(hidden_size, hidden_size))
			self.bn.append(self.get_bn_func(hidden_size))
		self.linears.append(nn.Linear(hidden_size, output_size))



	def forward(self, x):
		if len(x.shape) < 3:
			for i in range(self.num_layers - 1):
				if self.training and x.shape[0] == 1 and self.batch_norm:
					self.bn[i].eval()
					x = self.bn[i](self.linears[i](x))
					self.bn[i].train()
				else:
					x = self.bn[i](self.linears[i](x))
				x = self.relu(x)
			return self.linears[-1](x) # No activation function for output
		else:
			for i in range(self.num_layers - 1):
				x = self.linears[i](x)
				x = x.permute(0, 2, 1)
				if self.training and x.shape[0] == 1 and self.batch_norm:
					self.bn[i].eval()
					x = self.bn[i](x)
					self.bn[i].train()
				else:
					x = self.bn[i](x)
				x = x.permute(0, 2, 1)
				x = self.relu(x)
			return self.linears[-1](x) # No activation function for output
