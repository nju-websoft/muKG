import math

import torch
import torch.nn as nn
from .basic_model import BasicModel
import torch.nn.functional as F

from ...py.util.util import to_var


class DistMult(BasicModel):

	def __init__(self, kgs, args, dim = 100, margin = None, epsilon = None):
		super(DistMult, self).__init__(args, kgs)

		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.low_values = False
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		if self.args.init == 'xavier':
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		elif self.args.init == 'normal':
			std = 1.0 / math.sqrt(self.args.dim)
			nn.init.normal_(self.ent_embeddings.weight.data, 0, std)
			nn.init.normal_(self.rel_embeddings.weight.data, 0, std)
		elif self.args.init == 'uniform':
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor=self.ent_embeddings.weight.data,
				a=-self.embedding_range.item(),
				b=self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor=self.rel_embeddings.weight.data,
				a=-self.embedding_range.item(),
				b=self.embedding_range.item()
			)
		self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
		self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)

	def calc(self, h, r, t):
		score = (h * r) * t
		score = torch.sum(score, -1)
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		# mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = -self.calc(h, r, t).flatten()
		return score

	def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
		h_id = to_var(h_id, self.device)
		r_id = to_var(r_id, self.device)
		t_id = to_var(t_id, self.device)
		h_emb = self.ent_embeddings(h_id)
		t_emb = self.ent_embeddings(t_id)
		r_mat = self.rel_embeddings(r_id)
		b_size = len(h_id)
		if mode == 'entities':
			candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim)
			candidates = candidates.expand(b_size, self.ent_tot, self.dim)
		else:
			candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim)
			candidates = candidates.expand(b_size, self.rel_tot, self.dim)

		return h_emb, r_mat, t_emb, candidates

	def get_score(self, h, r, t):
		b_size = h.shape[0]
		if len(t.shape) == 3:
			assert (len(h.shape) == 2) & (len(r.shape) == 2)
			# this is the tail completion case in link prediction
			hr = (h * r).view(b_size, 1, self.dim)
			return -(hr * t).sum(dim=2)
		elif len(h.shape) == 3:
			assert (len(t.shape) == 2) & (len(r.shape) == 2)
			# this is the head completion case in link prediction
			rt = (r * t).view(b_size, 1, self.dim)
			return -(h * rt).sum(dim=2)
		elif len(r.shape) == 3:
			assert (len(h.shape) == 2) & (len(t.shape) == 2)
			# this is the relation prediction case
			hr = (h.view(b_size, 1, self.dim) * r)  # hr has shape (b_size, self.n_rel, self.emb_dim)
			return -(hr * t.view(b_size, 1, self.dim)).sum(dim=2)
