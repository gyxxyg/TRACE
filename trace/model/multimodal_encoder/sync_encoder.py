from transformers import PreTrainedTokenizer, AutoTokenizer
import re
import torch
import torch.nn as nn

class SyncTower(nn.Module):

    def __init__(self, hidden_dim=None):

        super().__init__()


        self.embed_tokens = nn.Embedding(1, hidden_dim)


    def forward(self, input_ids):
        input_ids = torch.zeros_like(input_ids).to(input_ids.device)
        tokens = self.embed_tokens(input_ids)
        return tokens 