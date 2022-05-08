import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_embeddings_torch(init, normalize, tot, dim, args):
    embeddings = nn.Embedding(tot, dim)
    if init == 'xavier':
        nn.init.xavier_uniform_(embeddings.weight.data)
        nn.init.xavier_uniform_(embeddings.weight.data)
    elif init == 'normal':
        std = 1.0 / math.sqrt(dim)
        nn.init.normal_(embeddings.weight.data, 0, std)
    elif init == 'uniform':
        embedding_range = nn.Parameter(
            torch.Tensor([(args.margin + args.epsilon) / dim]), requires_grad=False
        )
        nn.init.uniform_(
            tensor=embeddings.weight.data,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
    if normalize:
        embeddings.weight.data = F.normalize(embeddings.weight.data, 2, -1)
    return embeddings