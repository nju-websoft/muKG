import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.py.util.util import to_var
from src.torch.kge_models.basic_model import BasicModel


class TransE(BasicModel):

	def __init__(self, kgs, args, dim = 100, p_norm = 2, norm_flag = True, margin = 1.5, epsilon = None):
		super(TransE, self).__init__(args, kgs)
		self.dim = dim
		self.margin = margin
		# self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = 1
		self.margin_flag = False
		if self.args.loss == 'logistic_adv':
			self.margin_flag = True
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		#nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		#nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		self.epsilon = 2.0
		self.margin = nn.Parameter(
			torch.Tensor([self.args.margin]),
			requires_grad=False
		)
		self.embedding_range = nn.Parameter(
			torch.Tensor([(self.margin.item() + self.epsilon) / self.dim]),
			requires_grad=False
		)
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		nn.init.uniform_(tensor=self.ent_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
		nn.init.uniform_(tensor=self.rel_embeddings.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
		#self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
		#self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)

	def calc(self, h, r, t):
		score = (h + r) - t
		#score = torch.pow(torch.norm(score, self.p_norm, -1), 2)
		score = torch.norm(score, self.p_norm, -1)
		return score

	def get_embeddings(self, hid, rid, tid, mode='entity'):
		h = to_var(hid, self.device)
		r = to_var(rid, self.device)
		t = to_var(tid, self.device)
		r_embs = self.rel_embeddings(r).unsqueeze(1)
		proj_h = self.ent_embeddings(h).unsqueeze(1)  # shape: (b_size, 1, emb_dim)
		b_size = proj_h.shape[0]
		proj_t = self.ent_embeddings(t).unsqueeze(1)  # shape: (b_size, 1, emb_dim)
		if mode == 'entity':
			candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim)  # shape: (b_size, self.n_rel, self.emb_dim)
			candidates = candidates.expand(b_size, self.ent_tot, self.dim)
			return proj_h, r_embs, proj_t, candidates
		else:
			candidates = self.rel_embeddings.weight.data.view(1, self.rel_tot, self.dim)
			candidates = candidates.expand(b_size, self.rel_tot, self.dim)
			return proj_h, r_embs, proj_t, candidates

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		if self.margin_flag:
			score = self.calc(h, r, t).flatten() - self.margin
		else:
			score = self.calc(h, r, t).flatten()
		return score
