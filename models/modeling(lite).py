# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from timm.models.layers import trunc_normal_
import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 新增
def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError
# 新增
def get_index(idx,patch_num1=19,patch_num2=38):
    '''
    get index of fine stage corresponding to coarse stage 
    '''
    H1 = patch_num1
    H2 = patch_num2
    y = idx%H1
    idx1 = 4*idx - 2*y
    idx2 = idx1 + 1
    idx3 = idx1 + H2 
    idx4 = idx3 + 1 
    idx_finnal = torch.cat((idx1,idx2,idx3,idx4),dim=1)   # transformer对位置不敏感，位置随意
    return idx_finnal



class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Conv2d_BN(nn.Module):
    """Convolution with BN module."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, pad=0, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, act_layer=None):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        return x

# 有修改
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, act_layer=nn.Hardswish):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size1 = _pair(config.patches1["size"])
        patch_size2 = _pair(config.patches2["size"])
        self.average_pool = nn.AvgPool2d(2,stride=2)
        pad2 = 6    # patch_size=16，slide=12，块数量为38*38，这样能够保证边尺寸为上述的两倍，便于位置定位
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                    out_channels=config.hidden_size,
                                    kernel_size=patch_size2,
                                    stride=(config.slide_step, config.slide_step),
                                    padding=pad2)
        
        n_patches2 = ((img_size[0] + 2*pad2 - patch_size2[0]) // config.slide_step + 1) * ((img_size[0] + 2*pad2 - patch_size2[0]) // config.slide_step + 1)
        n_patches1 = int(n_patches2//4)
        self.position_embeddings1 = nn.Parameter(torch.zeros(1, n_patches1+1, config.hidden_size))
        self.position_embeddings2 = nn.Parameter(torch.zeros(1, n_patches2+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        embeddings1 = self.average_pool(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings2 = torch.cat((cls_tokens, x), dim=1)
        embeddings2 = embeddings2 + self.position_embeddings2
        embeddings2 = self.dropout(embeddings2)

        embeddings1 = embeddings1.flatten(2)
        embeddings1 = embeddings1.transpose(-1, -2)
        embeddings1 = torch.cat((cls_tokens, embeddings1), dim=1)
        embeddings1 = embeddings1 + self.position_embeddings1
        embeddings1 = self.dropout(embeddings1)

        return embeddings1, embeddings2

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            key_weight = np2th(weights[ROOT + "/" + ATTENTION_K + "/kernel"]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            value_weight = np2th(weights[ROOT + "/" + ATTENTION_V + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            out_weight = np2th(weights[ROOT + "/" + ATTENTION_OUT + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()

            query_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/bias"]).view(-1)
            key_bias = np2th(weights[ROOT + "/" + ATTENTION_K + "/bias"]).view(-1)
            value_bias = np2th(weights[ROOT + "/" + ATTENTION_V + "/bias"]).view(-1)
            out_bias = np2th(weights[ROOT + "/" + ATTENTION_OUT + "/bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + "/" + FC_0 + "/kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + "/" + FC_1 + "/kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + "/" + FC_0 + "/bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + "/" + FC_1 + "/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/bias"]))

        # 这份代码暂时会报key不存在的错误
        # ROOT = f"Transformer/encoderblock_{n_block}"
        # with torch.no_grad():
        #     query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

        #     query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
        #     key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
        #     value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
        #     out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

        #     self.attn.query.weight.copy_(query_weight)
        #     self.attn.key.weight.copy_(key_weight)
        #     self.attn.value.weight.copy_(value_weight)
        #     self.attn.out.weight.copy_(out_weight)
        #     self.attn.query.bias.copy_(query_bias)
        #     self.attn.key.bias.copy_(key_bias)
        #     self.attn.value.bias.copy_(value_bias)
        #     self.attn.out.bias.copy_(out_bias)

        #     mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
        #     mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
        #     mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
        #     mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

        #     self.ffn.fc1.weight.copy_(mlp_weight_0)
        #     self.ffn.fc2.weight.copy_(mlp_weight_1)
        #     self.ffn.fc1.bias.copy_(mlp_bias_0)
        #     self.ffn.fc2.bias.copy_(mlp_bias_1)

        #     self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        #     self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
        #     self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
        #     self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

# 有修改,把除vit以外的模块都加上cls前缀，方便加载预训练权重
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.cls_layer = nn.ModuleList()
        self.cls_numlayer = config.transformer["num_layers"]
        for _ in range(self.cls_numlayer):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        self.cls_norm = LayerNorm(config.hidden_size, eps=1e-6)


    def forward(self, hidden_states):
        attn_weights = []
        clss = []
        for i, layer in enumerate(self.layer):
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)   
            if i > 8:
                clss.append(self.cls_norm(hidden_states[:,0])) 

        return hidden_states, attn_weights, clss
# 有修改
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)
        self.target_index = [3,4,5,6,7,8,9,10,11]

    def forward(self, x):
        B = x.size(0)
        embedding_out1, embedding_out2 = self.embeddings(x)
        self.alpha = 0.4
        self.beta = 0.99
        global_attention = 0

        # 一阶段，将不重叠的patch(19*19)输入到encoder当中
        hidden_states1, weights1, _ = self.encoder(embedding_out1)
        import_token_num = math.ceil(self.alpha * 19 * 19)   
        # 分数整合及位置转换

        # # EMA
        # for index in range(len(weights1)):
        #     atten = weights1[index]
        #     if index in self.target_index:
        #         global_attention = self.beta*global_attention + (1-self.beta)*atten
        # cls_attn = global_attention.mean(dim=1)[:,0,1:] # (B,num_head,N,N)->(B,N,N)->(B,1,N-1) 相当于求cls和其他patch的相关性
        # policy_index = torch.argsort(cls_attn, dim=1, descending=True) # 把排序下标返回
        # print("policy_index",policy_index.shape)

        # random
        # policy_index = torch.randperm(19*19).unsqueeze(0)
        # policy_index = policy_index.expand(B, -1).cuda().long()
        # print("policy_index",policy_index.shape)

        # mean
        for index in range(len(weights1)):
            atten = weights1[index]
            if index in self.target_index:
                global_attention += atten
        cls_attn = global_attention.mean(dim=1)[:,0,1:] # (B,num_head,N,N)->(B,N,N)->(B,1,N-1) 相当于求cls和其他patch的相关性
        policy_index = torch.argsort(cls_attn, dim=1, descending=True) # 把排序下标返回
        # print("policy_index",policy_index.shape)

        important_index = policy_index[:, :import_token_num]
        important_index = get_index(important_index)
        # cls_index = torch.zeros((B,1)).long()
        cls_index = torch.zeros((B,1)).cuda().long()
        important_index = torch.cat((cls_index, important_index+1), dim=1)
        important_tokens = batch_index_select(embedding_out2, important_index)

        # 二阶段，将重叠patch(38*38)输入到encoder当中
        hidden_states2, attn_weights2, clss= self.encoder(important_tokens)
        
        return hidden_states1[:,0],clss

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.part_head1 = Linear(config.hidden_size, num_classes)
        self.part_head2 = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size*3),
            Linear(config.hidden_size*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            Linear(1024, num_classes),
        )

    def forward(self, x, labels=None):
        cls1, clss = self.transformer(x)
        final_cls = torch.cat((clss[-3], clss[-2], clss[-1]), dim=-1)
        logits1 = self.part_head1(cls1)
        logits2 = self.part_head2(final_cls)
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            part_loss = loss_fct(logits2.view(-1, self.num_classes), labels.view(-1))
            
            contrast_loss = con_loss(clss[-1], labels.view(-1))

            # x_log = F.log_softmax(logits1.view(-1, self.num_classes),dim=1)
            # y = F.softmax(logits2.view(-1, self.num_classes),dim=1)
            # kl = nn.KLDivLoss(reduction='batchmean')
            # loss3 = kl(x_log, y)

            loss3 = loss_fct(logits1.view(-1, self.num_classes), labels.view(-1))

            loss = part_loss + contrast_loss + loss3
            return loss, logits2
        else:
            return logits2

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.cls_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.cls_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings1
            
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings1.copy_(posemb)
            else:
                logging.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings1.copy_(np2th(posemb))
            # 参数复用
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new2 = self.transformer.embeddings.position_embeddings2
            if posemb.size() == posemb_new2.size():
                self.transformer.embeddings.position_embeddings2.copy_(posemb)
            else:
                logging.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new2.size()))
                ntok_new = posemb_new2.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings2.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('cls') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            # if self.transformer.embeddings.hybrid:
            #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
            #         for uname, unit in block.named_children():
            #             unit.load_from(weights, n_block=bname, n_unit=uname) 

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}



