from transformers import PreTrainedTokenizer, AutoTokenizer
import re
import torch
import torch.nn as nn

class TimeTower(nn.Module):

    def __init__(self, tokenizer, pretrain_tokenizer=None, hidden_dim=None, pretrain_embedding=None):

        super().__init__()

        # initialize tokenizer
        self.tokenizer = tokenizer
        # self.pretrain_tokenizer = pretrain_tokenizer

        # initialize token embedding weights
        assert hidden_dim is not None or pretrain_embedding is not None, 'can not set both pretrain embedding and hidden dim to None'
        # self.initialized = False
        if pretrain_embedding is not None:
            hidden_dim = pretrain_embedding.shape[1]
        self.embed_tokens = nn.Embedding(len(self.tokenizer.get_vocab()), hidden_dim)

        # use pretrained pretrain embedding to initialize weights
        # self.update_weights(pretrain_embedding)

    # def update_weights(self, pretrain_embedding):
    #     pretrain_tokenizer = self.pretrain_tokenizer
        
    #     if pretrain_tokenizer is not None and pretrain_embedding is not None:
    #         print(f'initializing time tower weights {pretrain_embedding}')
    #         self.initialized = True

    #         pretrain_token_ids = self._get_pretrain_tokenizer_ids(pretrain_tokenizer)
    #         vocab = self.tokenizer.get_vocab()

    #         # copy the pretrained weights
    #         for key, ids in pretrain_token_ids.items():
    #             weights = torch.mean(pretrain_embedding[ids], dim=0)
    #             self.embed_tokens.weight.data[vocab[key]] = weights

    def _get_pretrain_tokenizer_ids(self, pretrain_tokenizer):

        pretrain_token_ids = {k: pretrain_tokenizer(k).input_ids for k in self.tokenizer.get_vocab()}

        # remove bos token
        for k, v in pretrain_token_ids.items():
            if v[0] == pretrain_tokenizer.bos_token_id:
                pretrain_token_ids[k] = v[1:]

        return pretrain_token_ids

    def encode(self, timestamps):

        def insert_separator(X, sep):
            return [self.tokenizer(ele).input_ids for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        
        # format to the same length
        timestamps = [format(t, '0>6.1f') for t in timestamps]

        # get input ids
        input_ids = []
        for ids in insert_separator(timestamps, '<sep>'):
            input_ids.extend(ids)
        input_ids.extend(self.tokenizer('<sync>').input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # print(input_ids)

        return input_ids

        # # encode
        # tokens = self.embed_tokens(input_ids)

        # return tokens   
    def forward(self, input_ids):
        tokens = self.embed_tokens(input_ids)
        return tokens  



class TimeTokenizer(PreTrainedTokenizer):
    def __init__(self, *args, **kwargs):
        self.vocab = {}
        self.vocab['<sync>'] = 0
        self.vocab['<sep>'] = 1
        for i in range(10):
            self.vocab[str(i)] = i + 2
        self.vocab['.'] = 10 + 2
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(*args, **kwargs)

    def _tokenize(self, text, *args, **kwargs):
        pattern = '|'.join(map(re.escape, self.vocab.keys()))
        tokens = re.findall(pattern, text)
        # print(tokens)
        return [token for token in tokens if token]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab.keys())



if __name__ == "__main__":

    # pretrain_tokenizer = AutoTokenizer.from_pretrained('/group/40065/model/vicuna-7b-v1.5/', use_fast=False)

    # print(pretrain_tokenizer(['12sfgh sfjjs2345']))

    # print(pretrain_tokenizer.bos_token_id)

    # print(pretrain_tokenizer.decode([18668, 29871]))

    tokenizer = TimeTokenizer()

    pretrain_embedding = torch.rand(32000, 10)

    pretrain_tokenizer = AutoTokenizer.from_pretrained('/group/40065/model/vicuna-7b-v1.5/', use_fast=False)

    time_encoder = TimeTower(tokenizer, pretrain_tokenizer=pretrain_tokenizer, pretrain_embedding=pretrain_embedding)

    print(torch.cat([time_encoder([12.3, 45.6]), time_encoder([12.3, 45.6]), time_encoder([12.3]), time_encoder([])], dim=0).shape)

