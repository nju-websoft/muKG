import torch
import torch.nn as nn
# from .Model import Model
import numpy
from numpy import fft
import torch.nn.functional as F
from src.torch.kge_models.basic_model import BasicModel
from ...py.util.util import to_var


class HolE_ET(BasicModel):

	def __init__(self, args, kgs, dim = 100, margin = None, epsilon = None):
		super(HolE_ET, self).__init__(args, kgs)

		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.low_values = False
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(1, self.dim)
		self.type_embeddings = nn.Embedding(3851, self.dim)
		
		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.type_embeddings.weight.data)
		self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
		self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)
		self.type_embeddings.weight.data = F.normalize(self.type_embeddings.weight.data, 2, -1)

	def _ccorr(self, r):
		b_size, dim = r.shape
		x = r.view(b_size, 1, dim)
		return torch.cat([x.roll(i, dims=2) for i in range(dim)], dim=1)

	def calc(self, h, r, t):
		r = self._ccorr(r)
		#print(r.shape)
		hr = torch.matmul(h.view(-1, 1, self.dim), r)
		return -(hr.view(-1, self.dim) * t).sum(dim=1)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = F.normalize(self.ent_embeddings(batch_h), 2, -1)
		t = F.normalize(self.type_embeddings(batch_t), 2, -1)
		r = self.rel_embeddings(batch_r)
		#print(r.shape)
		score = self.calc(h, r, t)
		return score

	def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
		h_id = to_var(h_id, self.device)
		r_id = to_var(r_id, self.device)
		t_id = to_var(t_id, self.device)
		h_emb = self.ent_embeddings(h_id)
		t_emb = self.type_embeddings(t_id)
		r_mat = self._ccorr(self.rel_embeddings(r_id))
		b_size = len(h_id)
		if mode == 'entities':
			candidates = self.type_embeddings.weight.data.view(1, 3851, self.dim)
			candidates = candidates.expand(b_size, 3851, self.dim)
		else:
			r_mat = self._ccorr(self.rel_embeddings.weight.data)
			candidates = r_mat.view(1, self.rel_tot, self.dim, self.dim)
			candidates = candidates.expand(b_size, self.rel_tot, self.dim, self.dim)

		return h_emb, r_mat, t_emb, candidates

	def get_score(self, h, r, t):
		b_size = h.shape[0]
		if len(t.shape) == 3:
			assert (len(h.shape) == 2) & (len(r.shape) == 3)
			# this is the tail completion case in link prediction
			h = h.view(b_size, 1, self.dim)
			hr = torch.matmul(h, r).view(b_size, self.dim, 1)
			return -(hr * t.transpose(1, 2)).sum(dim=1)
		elif len(h.shape) == 3:
			assert (len(t.shape) == 2) & (len(r.shape) == 3)
			# this is the head completion case in link prediction
			t = t.view(b_size, self.dim, 1)
			return -(h.transpose(1, 2) * torch.matmul(r, t)).sum(dim=1)
		elif len(r.shape) == 4:
			assert (len(h.shape) == 2) & (len(t.shape) == 2)
			# this is the relation completion case in link prediction
			h = h.view(b_size, 1, 1, self.dim)
			t = t.view(b_size, 1, self.dim)
			hr = torch.matmul(h, r).view(b_size, self.rel_tot, self.dim)
			return -(hr * t).sum(dim=2)

