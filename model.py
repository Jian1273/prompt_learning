import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

class FcPromptModel(nn.Module):
    def __init__(self, model_path, num_class, mask_index):
        super(FcPromptModel, self).__init__()
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(self.model_path, num_labels=num_class)
        self.bert = AutoModel.from_pretrained(self.model_path, config=self.config)
        self.num_class = num_class
        self.mask_index = mask_index
        self.fc = nn.Linear(self.config.hidden_size, self.num_class)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:,self.mask_index,:]
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x

class FineTuneModel(nn.Module):
    def __init__(self, model_path, num_class):
        super(FineTuneModel, self).__init__()
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(self.model_path, num_labels=num_class)
        self.bert = AutoModel.from_pretrained(self.model_path, config=self.config)
        self.num_class = num_class
        self.fc = nn.Linear(self.config.hidden_size, self.num_class)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.fc(x['pooler_output'])
        # x = F.softmax(x, dim=1)
        return x

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)

class SoftPromptModel(nn.Module):
    def __init__(self, model_path, num_class, n_tokens):
        super(SoftPromptModel, self).__init__()
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(self.model_path, num_labels=num_class)
        self.model = AutoModel.from_pretrained(self.model_path, config=self.config)
        self.s_wte = SoftEmbedding(self.model.get_input_embeddings(), n_tokens=10, initialize_from_vocab=True)
        self.model.set_input_embeddings(self.s_wte)
        self.num_class = num_class
        self.n_tokens = n_tokens
        self.fc = nn.Linear(self.config.hidden_size, self.num_class)

    def forward(self, input_ids, attention_mask):
        input_ids = torch.cat([torch.full((input_ids.shape[0],self.n_tokens), 10086).cuda(), input_ids], 1)
        attention_mask = torch.cat([torch.full((attention_mask.shape[0],self.n_tokens), 1).cuda(), attention_mask], 1).cuda()
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.fc(x['pooler_output'])
        # x = F.softmax(x, dim=1)
        return x

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
