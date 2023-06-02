from __future__ import absolute_import, division, print_function 
import os, copy, logging, math, random, time, typing, io, cv2
import numpy as np 
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import albumentations as albu
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import ndimage

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torchvision
from torchvision import transforms, datasets
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import utils
import torch.nn.functional as F

from matplotlib.colors import LogNorm 
from itertools import combinations, permutations
from urllib.request import urlretrieve
from PIL import Image
from os.path import join as pjoin

from tqdm import tqdm
from datetime import timedelta
# from apex import amp
import ml_collections
import sys
from thop import profile
sys.path.append("/workspace/CICC-main/experiments train - using/")
sys.path.append("/workspace/CICC-main/experiments train - using/Torch-Pruning/")
sys.path.append("/workspace/CICC-main/experiments train - using/Torch-Pruning/torch_pruning")
from torchsummary import summary
from torchpruner.attributions import ShapleyAttributionMetric,TaylorAttributionMetric,WeightNormAttributionMetric,SensitivityAttributionMetric,APoZAttributionMetric
import experiments.models.cifar10 as cifar10
from torchpruner.pruner import Pruner
import torch_pruning as tp
import netron
import torch.onnx

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('GPU: ', torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print('No GPU available')
test_mlp_dim = 512

def get_testing():
    '''
    Returns a minimal configuration for testing
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_b16_config():
    '''
    Returns the ViT-B/16 configuration
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = test_mlp_dim
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_b32_config():
    '''
    Returns the ViT-B/32 configuration
    '''
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config

def get_l16_config():
    '''
    Returns the ViT-L/16 configuration
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_l32_config():
    '''
    Returns the ViT-L/32 configuration
    '''
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config

def get_h14_config():
    '''
    Returns the ViT-L/16 configuration
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights):
    """Possibly convert HWIO to OIHW."""
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}




class Attention(nn.Module):
    head_value = False
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        
        self.vis = vis
        if self.head_value == False:
            self.num_attention_heads = config.transformer["num_heads"]
            self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        else:
            self.num_attention_heads = 8
            self.attention_head_size = 64
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size   #剪8个头 4*64=256

        self.query = Linear(config.hidden_size, self.all_head_size)                #剪768 512 [256,197,256]
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        a = x.size()[:-1]
        if self.head_value == False:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #[256,197,12,64]
        else:
            new_x_shape = x.size()[:-1] + (8, self.attention_head_size) #[256,197,12,64]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)    # [batch,197,768]          #剪[batch,197,512]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [batch,12,197,64]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   # [batch,12,197,197]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [batch,12,197,197]
        attention_probs = self.softmax(attention_scores)                            # [batch,12,197,197]
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)                  # [batch,12,197,64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()              # [batch,197,12,64]
        if self.head_value == False:
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [batch,197,768]
        else:
            new_context_layer_shape = context_layer.size()[:-2] + (512,) # [batch,197,768]
        context_layer = context_layer.view(*new_context_layer_shape)                # [batch,197,768]
        attention_output = self.out(context_layer)                                  # [batch,197,768]
        attention_output = self.proj_dropout(attention_output)                      # [batch,197,768]
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


class Embeddings(nn.Module):
    '''
    Construct the embeddings from patch, position embeddings
    '''
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):                # [256,3,224,224]
        B = x.shape[0]                   # 256
        cls_tokens = self.cls_token.expand(B, -1, -1) #[256,1,768]

        x = self.patch_embeddings(x)     #[256,768,14,14]
        x = x.flatten(2)                 #[256,768,196]
        x = x.transpose(-1, -2)          #[256,196,768]
        x = torch.cat((cls_tokens, x), dim=1)  #[256,197,768]

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)      #[256,197,768]
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()[:test_mlp_dim, :768]
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()[:768, :test_mlp_dim]
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()[:test_mlp_dim]
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()[:768]

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
 

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=10, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)

        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights
        

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"]))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
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
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'testing': get_testing(),
}

config = CONFIGS["ViT-B_16"]
# model = VisionTransformer(config, img_size=224, num_classes=21843, zero_head=False, vis=True)
# model.load_from(np.load("/workspace/CICC-main/experiments train - using/Vision-Transformer-main/ViT-B_16.npz"))
# model.to(device)

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_loader(local_rank, img_size, dataset, train_batch_size, eval_batch_size):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dataset == "cifar10":
        trainset = datasets.CIFAR10(root="/workspace/CICC-main/cifar10-vgg16/data.cifar10",
                                    train=True,
                                    download=False,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="/workspace/CICC-main/cifar10-vgg16/data.cifar10",
                                   train=False,
                                   download=False,
                                   transform=transform_test) if local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=False,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=False,
                                    transform=transform_test) if local_rank in [-1, 0] else None
    if local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(output_dir, name, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_checkpoint = os.path.join(output_dir, "%s_checkpoint.bin" % name)
    # torch.save(model_to_save.state_dict(), model_checkpoint)
    model_checkpoint = os.path.join(output_dir, "%s_checkpoint.pth" % name)           #1剪枝一个linear #2剪枝所有linear #3剪枝完所有的训练400次 #4prune_head剪枝head未训练 #5训练500次 #训练600次
    torch.save(model_to_save, model_checkpoint)                                        #
    logger.info("Saved model checkpoint to [DIR: %s]", output_dir)


def setup(model_type, img_size, pretrained_dir, device, dataset):
    # Prepare model
    config = CONFIGS[model_type]

    num_classes = 10 if dataset == "cifar10" else 100

    model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(pretrained_dir))
    model = torch.load("/workspace/CICC-main/experiments train - using/main_net/output_cifar10_1000_original_all_formal_warm=300_lr_0.03/cifar10_1000_all_formal1_warm=300_lr=0.03_checkpoint.pth")
    # model = torch.load("/workspace/CICC-main/experiments train - using/main_net/output_cifar10_1000_original_all_formal_warm=300_lr_0.03/cifar10_1000_all_formal1_warm=300_lr=0.03_checkpoint.pth")
    # model = torch.load("/workspace/CICC-main/experiments train - using/main_net/analyze/512_pre_checkpoint.pth")
    model.eval()
    model.to(device)
    input = torch.randn(1, 3, 224, 224).cuda()
    macs, params = profile(model, inputs=(input,))
    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", model_type, img_size, pretrained_dir, device)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    logger.info("FLOPs:%sG,params:%sM",macs/2000000000,params/1000000)
    
    return model_type, img_size, pretrained_dir, device, model
    

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def easy_valid(model,test_loader):
    eval_losses = AverageMeter()
    eval_batch_size = 64
    local_rank =-1
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, attn_weights = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    # logger.info("\n")
    # logger.info("Validation Results")
    # logger.info("Global Steps: %d" % global_step)
    # logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    # logger.info("Valid Accuracy: %2.5f" % accuracy)

    # writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy

def valid(eval_batch_size, local_rank, device, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, attn_weights = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    return accuracy

def loss_P(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model,X_train=None):
    """ Train the model """

    if local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", name))
    
    train_batch_size = train_batch_size // gradient_accumulation_steps
    
   
    # Prepare dataset
    train_loader, test_loader = get_loader(local_rank, img_size, dataset, train_batch_size, eval_batch_size)
    if(X_train!=None):
        train_loader = X_train
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    t_total = num_steps
    if decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # if fp16:
        # model, optimizer = amp.initialize(models=model,
        #                                   optimizers=optimizer,
        #                                   opt_level=fp16_opt_level)
        # amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    # if local_rank != -1:
        # model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    # print(model)
    # summary(model, input_size=(3, 224, 224), device='cuda')
    model.zero_grad()
    set_seed(seed, n_gpu)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    train_accs = []
    test_accs = []

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            # if fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                losses.update(loss.item()*gradient_accumulation_steps)
                if fp16:
                    empty = empty
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)


                if global_step % eval_every == 0 and local_rank in [-1, 0]:     # 每一百个测一下准确率
                    accuracy = valid(eval_batch_size, local_rank, device, model, writer, test_loader, global_step)
                    test_accs.append(accuracy)
                    if best_acc < accuracy:
                        save_model(output_dir, name, model)
                        best_acc = accuracy
                    model.train()

                if global_step % 5 == 0:    # 每训练5次保存一次
                    torch.save({
                        'epoch': global_step,
                        'model': model,
                        'optimizer': optimizer,
                        'scheduler': scheduler,
                        'test_acc': test_accs
                    }, './checkpoint.pt')
                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
    input = torch.randn(1, 3, 224, 224).cuda()
    macs, params = profile(model, inputs=(input,))
    writer.add_scalar("model/flops", scalar_value=macs/2000000000)
    writer.add_scalar("model/params", scalar_value=params/1000000)
    if local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

def mlp_layer_prune(model,loss,val_loader,device):
    cal_count = 0
    attr = ShapleyAttributionMetric(model, val_loader, loss, device, sv_samples=5)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    module1 = module
                                                    scores = attr.run(module1)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))

    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/', '512_mlp_5.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
np.array(res, dtype=object))
    return res
                                                    
                                                    
    # np.save(os.path.join('./trained_shap_values', 'shap_value.npy'),
    #     np.array(res, dtype=object))

def assign_prune_channels(prune_channels: list,
                          num_block):
    for i in range(num_block):  

            prune_channels[i] = i
            
def get_channel_index(num_elimination, sv):

    sv = sv.astype(float)
    sv = torch.from_numpy(sv)
    linear_count = 0
    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    vals, indices = torch.sort(sv)
    # for i in vals:
    #     if i < 0:
    #         linear_count +=1
    # return indices[:linear_count].tolist()
    # return indices[:1430].tolist()
    return indices[:1430].tolist()

def get_head_index(num_elimination, sv):
    """
    kernel: network.features[i]里的Conv模块
    num_elimination: prune-channels数组: [1 1] => 保留排序后的前几个，此处保留第一个
    get cadidate channel index for pruning，按weight排序，返回截取后的索引
    获取用于修剪的候选通道索引
    """
    sv = sv.astype(float)
    sv = torch.from_numpy(sv)
    head_list = []
    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    vals, indices = torch.sort(sv)
    head_prune = list(list(combinations([0,1, 2, 3,4,5,6,7,8,9,10,11], 8))[indices[0]])   #要减去head的数量
    for ii in range(8):
        head_count_value = head_prune[ii]*64 + 64
        for iii in range(head_count_value-64,head_count_value,1):
            head_list.append(iii)
    # for i in vals:
    #     if i < 0:
    #         num_elimination +=1
    # return indices[:num_elimination].tolist()
    return head_list

def mlp_prune_f(model):
    prune_channels = [0] * 12
    assign_prune_channels(prune_channels,
                              num_block=1)
    shap_value = np.load("/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/sv_5_new_formal_mlp_shap_value_1000.npy", allow_pickle=True)
    # assert len(prune_channels) == len(shap_value)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224).cuda())

    # for module in model.modules():
    #     if isinstance(module, VisionTransformer):
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    if(prune_count==12):
                                        break
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    channel_index = get_channel_index(prune_channels[prune_count], shap_value[prune_count])

                                                    pruning_plan = DG.get_pruning_plan(module, tp.prune_linear_out_channels, idxs=channel_index)
                                                    pruning_plan.exec()
                                                    # print('prune_count:{}'.format(prune_count))
                                                    # print(model)
                                                    # model=train_iterative(model)
                                                    prune_count += 1  # only prune fc1
                                                    if(prune_count==12):
                                                        print(model)
                                                        break

    return model

def talor_mlp_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = TaylorAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
#     output_dir = '/workspace/CICC-main/experiments train - using/main_net/talor_value/'        
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/talor_value/','1024_talor_mlp.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
# np.array(res, dtype=object))
    return res

def talor_head_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = TaylorAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
    output_dir = '/workspace/CICC-main/experiments train - using/main_net/talor_value/'        
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/talor_value/','4head_talor_value.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
np.array(res, dtype=object))


def weight_mlp_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = WeightNormAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
#     output_dir = '/workspace/CICC-main/experiments train - using/main_net/weight_value/'        
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/weight_value/','1024_weight_mlp.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
# np.array(res, dtype=object))
    return res

def weight_head_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = WeightNormAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
    output_dir = '/workspace/CICC-main/experiments train - using/main_net/weight_value/'        
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/weight_value/','4head_weight_value.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
np.array(res, dtype=object))


def apoz_mlp_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = APoZAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
#     output_dir = '/workspace/CICC-main/experiments train - using/main_net/apoz_value/'        
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/apoz_value/','1024_apoz_mlp.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
# np.array(res, dtype=object))
    return res
    
def apoz_head_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = APoZAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
    output_dir = '/workspace/CICC-main/experiments train - using/main_net/apoz_value/'        
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/apoz_value/','4head_apoz.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
    np.array(res, dtype=object))


def sens_mlp_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = SensitivityAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
#     output_dir = '/workspace/CICC-main/experiments train - using/main_net/sens_value/trained_weight_values/'        
#     os.makedirs(output_dir, exist_ok=True)
#     np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/sens_value/trained_weight_values','formal_mlp_talor_value_1000.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
# np.array(res, dtype=object))
    return res

def sens_head_cal(model,loss,val_loader,device):
    cal_count = 0
    attr = SensitivityAttributionMetric(model, val_loader, loss, device)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    scores = attr.run(module)
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
    output_dir = '/workspace/CICC-main/experiments train - using/main_net/sens_value/'        
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/sens_value/','4head_sens.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算head剩下4个头
np.array(res, dtype=object))
    return res


def get_talor_index(num_elimination, sv):

    sv = sv.astype(float)
    sv = torch.from_numpy(sv)
    linear_count = 0
    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    vals, indices = torch.sort(sv)
    # for i in vals:
    #     if i < 0:
    #         linear_count +=1
    return indices[:448].tolist()

def get_random_index(num_elimination, sv):

    sv = sv.astype(float)
    sv = torch.from_numpy(sv)
    linear_count = 0
    # ori: [4, 2, 3, 1] => values=tensor([1, 2, 3, 4]), indices=tensor([3, 1, 2, 0]))
    # 绝对值排序，按照最小值挑出前num_elimination个的下标
    # vals, indices = torch.sort(sv)
    indices = np.random.permutation(3072)
    # for i in vals:
    #     if i < 0:
    #         linear_count +=1
    return indices[:1430].tolist()


def any_mlp_prune(model,res=None):
    prune_channels = [0] * 12
    assign_prune_channels(prune_channels,
                              num_block=1)
    any_value = np.load("/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/512_mlp.npy", allow_pickle=True)
    # assert len(prune_channels) == len(shap_value)
    # any_value = res
    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224).cuda())

    # for module in model.modules():
    #     if isinstance(module, VisionTransformer):
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    if(prune_count==12):
                                        break
                                    for name, module in module.named_children():
                                        if name in ['ffn']:
                                            for name, module in module.named_children():
                                                if name in ['fc1']:
                                                    channel_index = get_talor_index(prune_channels[prune_count], any_value[prune_count])

                                                    pruning_plan = DG.get_pruning_plan(module, tp.prune_linear_out_channels, idxs=channel_index)
                                                    pruning_plan.exec()
                                                    print(model)
                                                    prune_count += 1  # only prune fc1
                                                    if(prune_count==12):
                                                        break

    return model

def get_any_head_index(sv):

    sv = sv.astype(float)
    sv = torch.from_numpy(sv)
    head_count = 1
    inner_value = []
    head_value = []
    head_list = []
    j=0

    for i in range(768):
        inner_value.append(sv[i])
        if (i+1)%64==0 :
            head_value.append(sum(inner_value)/64)
            j += 1
            inner_value = []
    indices = np.argsort(head_value)
    for ii in range(4):
        head_count_value = indices[ii]*64 + 64
        for iii in range(head_count_value-64,head_count_value,1):
            head_list.append(iii)

    return head_list[:256]


def any_head_prune(model):
    prune_channels = [0] * 12
    assign_prune_channels(prune_channels,
                              num_block=1)
    any_value = np.load("/workspace/CICC-main/experiments train - using/main_net/sens_value/4head_sens.npy", allow_pickle=True)
    # assert len(prune_channels) == len(shap_value)
    # any_value = res
    prune_count = 0
    channel_index=[]
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224).cuda())

    # for module in model.modules():
    #     if isinstance(module, VisionTransformer):
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    if(prune_count==12):
                                        break
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    channel_index = get_any_head_index(any_value[prune_count])

                                                    pruning_plan = DG.get_pruning_plan(module, tp.prune_linear_out_channels, idxs=channel_index)
                                                    pruning_plan.exec()
                                                    
                                                    prune_count += 1  # only prune fc1
                                                    if(prune_count==12):
                                                        output="/workspace/CICC-main/experiments train - using/main_net/sens/"
                                                        os.makedirs(output, exist_ok=True)
                                                        save_model(output,'split_sens_4head_not_train',model)
                                                        print(model)
                                                        break

    return model


def head_prune_f(model):
    prune_channels = [0] * 12
    assign_prune_channels(prune_channels,
                              num_block=1)
    shap_value = np.load("/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/shap_value_1000_prune_8_head.npy", allow_pickle=True)
    # assert len(prune_channels) == len(shap_value)

    prune_count = 0

    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224).cuda())

    # for module in model.modules():
    #     if isinstance(module, VisionTransformer):
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    if(prune_count==12):
                                        break
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    channel_index = get_head_index(prune_channels[prune_count], shap_value[prune_count])

                                                    pruning_plan = DG.get_pruning_plan(module, tp.prune_linear_out_channels, idxs=channel_index)
                                                    pruning_plan.exec()
                                                    
                                                    prune_count += 1  # only prune fc1
                                                    if(prune_count==12):
                                                        print(model)
                                                        break

    return model


def shapley_calculate_head_f(model,loss,val_loader,device):
    cal_count = 0
    attr = ShapleyAttributionMetric(model, val_loader, loss, device, sv_samples=1)
    # layer_mlp = list()
    res = []
    for name, module in model.named_children():
        if name in ['transformer']:
            for name, module in module.named_children():
                if name in ['encoder']:
                    for name, module in module.named_children():
                        if name in ['layer']:
                            for name, module in module.named_children():
                                if isinstance(name,str):
                                    for name, module in module.named_children():
                                        if name in ['attn']:
                                            for name, module in module.named_children():
                                                if name in ['query']:
                                                    scores = attr.run(module)[:495] 
                                                    res.append(scores)
                                                    cal_count += 1
                                                    print("cal_count{}".format(cal_count))
    np.save(os.path.join('/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/', 'shap_value_1000_prune_4_head.npy'),  #_1是只计算了第一个层的 _2是计算了12个层的 _3计算所有head
np.array(res, dtype=object))
                                                    


def test_module_f(model):
    images = torch.zeros(1, 3, 224, 224).to(device).cuda()
    for i in range(5):
      output = model(images)  
    return output


def sv_read():
    prune_channels = [0] * 12
    assign_prune_channels(prune_channels,
                              num_block=1)
    shap_value = np.load("/workspace/CICC-main/experiments train - using/main_net/trained_shap_values/shap_value_1000_prune_4_head.npy", allow_pickle=True)
    # assert len(prune_channels) == len(shap_value)

    prune_count = 0
    for i in range(12):
        sv=shap_value[prune_count]
        sv = sv.astype(float)
        sv = torch.from_numpy(sv)
        # print("number:{},{}".format(i,sv))
        # sv=sum(sv)/len(sv)
        vals, indices = torch.sort(sv)

        print("4head最小值:{}".format(vals[0]))
        prune_count += 1  # only prune fc1

import torch.distributed as dist
import os
import argparse
from sklearn.model_selection import KFold
# 嵌套交叉验证的代码
def cross_validate(model,way,batch):
    kfold_out=KFold(n_splits = 5,shuffle = True)
    kfold_in =KFold(n_splits = 3,shuffle = True) #内部嵌套
    X,V= cifar10.get_dataset()
    accuracies = []
    loss = loss_P
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    res = 0
    out_count = []
    in_count = []
    for i in range(len(X)):
        out_count.append(i)
    for train_index, test_index in kfold_out.split(out_count):
        # 划分训练集和测试集
        X_train, X_test = torch.utils.data.Subset(X,train_index),torch.utils.data.Subset(X,test_index)

        train_sampler = RandomSampler(X_train) 
        test_sampler = SequentialSampler(X_test)
        train_out_loader = DataLoader(X_train,
                            sampler=train_sampler,
                            batch_size=batch,
                            num_workers=8,
                            pin_memory=True)
        test_out_loader = DataLoader(X_test,
                            sampler=test_sampler,
                            batch_size=batch,
                            num_workers=8,
                            pin_memory=True) 
        
        for i in range(len(X_train)):
            in_count.append(i)
        for inner_train_index , inner_val_index in kfold_in.split(in_count):
            # 在子训练集上拟合模型

            X_in_train, X_in_validate = torch.utils.data.Subset(X_train,inner_train_index),torch.utils.data.Subset(X_train,inner_val_index)
            train_sampler = RandomSampler(X_in_train) 
            test_sampler = SequentialSampler(X_in_validate)
            train_inner_loader = DataLoader(X_in_train,
                              sampler=train_sampler,
                              batch_size=batch,
                              num_workers=8,
                              pin_memory=True)
            test_inner_loader = DataLoader(X_in_validate,
                             sampler=test_sampler,
                             batch_size=batch,
                             num_workers=8,
                             pin_memory=True) 
            
            train_iterative(model,train_inner_loader)
            # 使用验证集计算重要性
            if (way == 'sens'):
                res =+ sens_mlp_cal(model,loss,test_inner_loader,device)  
            elif(way == 'talor'):
                res =+ talor_mlp_cal(model,loss,test_inner_loader,device)
            elif(way == 'apoz'):
                res =+ apoz_mlp_cal(model,loss,test_inner_loader,device)
            else:
                raise ValueError("Invalid pruning method specified.")
        res /= kfold_in.n_splits
        # 剪枝
        model = any_mlp_prune(model,res)
        #剪枝后训练
        train_iterative(model,train_out_loader)
        # 在测试集上进行预测
        accuracy=easy_valid(model,test_out_loader)
        # 将准确率添加到列表中
        accuracies.append(accuracy)
    for fold, accuracy in enumerate(accuracies):
        print(f"Fold {fold+1} Accuracy: {accuracy}")
    
    mean_accuracy = np.mean(np.array(accuracies))
    print("{} Mean Accuracy: {}".format(way,mean_accuracy))

def mean_normalization(value):
    value = (value-np.min(value))/(np.max(value)-np.min(value))
    return value

def weigh_prune():
    sens_weigh   = 0.35
    taylor_weigh = 0.35
    apoz_weigh   = 0.3

    sense_value =  np.load("/workspace/CICC-main/experiments train - using/main_net/sens_value/trained_weight_values/formal_mlp_talor_value_1000.npy", allow_pickle=True)
    taylor_value=  np.load("/workspace/CICC-main/experiments train - using/main_net/talor_value/trained_talor_values/formal_mlp_talor_value_1000.npy", allow_pickle=True)
    apoz_value  =  np.load("/workspace/CICC-main/experiments train - using/main_net/apoz_value/trained_weight_values/formal_mlp_talor_value_1000.npy", allow_pickle=True)
  
    sense_value = mean_normalization(sense_value)
    taylor_value= mean_normalization(taylor_value)
    apoz_value  = mean_normalization(apoz_value)

    weigh_importance = sens_weigh*sense_value+taylor_weigh*taylor_value+apoz_weigh*apoz_value
    return weigh_importance



def train_iterative(model,X_train):
    name = "cross_value"                                 # Name of this run. Used for monitoring
    dataset = "cifar10"                                 # Which downstream task, choices=["cifar10", "cifar100"]
    model_type = "ViT-B_16"                             # Which variant to use
    pretrained_dir = "/workspace/CICC-main/experiments train - using/Vision-Transformer-main/ViT-B_16.npz"            # Where to search for pretrained ViT models
    output_dir = "/workspace/CICC-main/experiments train - using/main_net/iterative/mlp_prune_1430__train"                     # The output directory where checkpoints will be written
    img_size = 224                                      # Resolution size
    train_batch_size = 128                               # Total batch size for training
    eval_batch_size = 128                                # Total batch size for eval
    eval_every = 100                                    # Run prediction on validation set every so many steps (Will always run one evaluation at the end of training)
    learning_rate = 5e-2                                # The initial learning rate for SGD
    weight_decay = 0                                    # Weight deay if we apply some
    num_steps = 500                                     # Total number of training epochs to perform 500
    decay_type = "cosine"                               # How to decay the learning rate, choices=["cosine", "linear"]
    warmup_steps = 300                                  # Step of training to perform learning rate warmup for
    max_grad_norm = 1.0                                 # Max gradient norm
    seed = 42                                           # random seed for initialization  42
    gradient_accumulation_steps = 1                     # Number of updates steps to accumulate before performing a backward/update pass
    fp16 = 0                                            # (action = 'store_true') Whether to use 16-bit float precision instead of 32-bit
    fp16_opt_level = 'store_true'                       # For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']
    loss_scale = 0                                      # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True (0 (default value): dynamic loss scaling, Positive power of 2: static loss scaling value)
    local_rank = -1                                     # local_rank for distributed training on gpus
    device="cpu"
    n_gpu = 0
    iterative_prune = True
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (local_rank, device, n_gpu, bool(local_rank != -1), fp16))
    

    set_seed(seed, n_gpu)
    # print('using_gpu{}'.format(n_gpu))
    # Model & Tokenizer Setup
    model_type, img_size, pretrained_dir, device, _ = setup(model_type, img_size, pretrained_dir, device, dataset)

    train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model,X_train)
    return model


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Required parameters
    name = "shapley_1000_sv_5_mlp_prune_1e-2"                                 # Name of this run. Used for monitoring
    dataset = "cifar10"                                 # Which downstream task, choices=["cifar10", "cifar100"]
    model_type = "ViT-B_16"                             # Which variant to use
    pretrained_dir = "/workspace/CICC-main/experiments train - using/Vision-Transformer-main/ViT-B_16.npz"            # Where to search for pretrained ViT models
    output_dir = "/workspace/CICC-main/experiments train - using/main_net/shapley/"                     # The output directory where checkpoints will be written
    img_size = 224                                      # Resolution size
    train_batch_size = 64                               # Total batch size for training
    eval_batch_size = 64                                # Total batch size for eval
    eval_every = 100                                    # Run prediction on validation set every so many steps (Will always run one evaluation at the end of training)
    learning_rate = 1e-2                                # The initial learning rate for SGD
    weight_decay = 0                                    # Weight deay if we apply some
    num_steps = 1000                                     # Total number of training epochs to perform 500
    decay_type = "cosine"                               # How to decay the learning rate, choices=["cosine", "linear"]
    warmup_steps = 300                                  # Step of training to perform learning rate warmup for
    max_grad_norm = 1.0                                 # Max gradient norm
    seed = 42                                           # random seed for initialization  42
    gradient_accumulation_steps = 1                     # Number of updates steps to accumulate before performing a backward/update pass
    fp16 = 0                                            # (action = 'store_true') Whether to use 16-bit float precision instead of 32-bit
    fp16_opt_level = 'store_true'                       # For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']
    loss_scale = 0                                      # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True (0 (default value): dynamic loss scaling, Positive power of 2: static loss scaling value)
    local_rank = -1                                     # local_rank for distributed training on gpus
    device="cpu"
    n_gpu = 0
    shapley_calculate = False                        #是否进行计算value
    mlp_prune =   True                                 #是否进行剪枝
    test_module = False
    shapley_calculate_head = False
    head_prune = False
    
    talor_mlp_prune_f = False
    talor_mlp_prune_v = False
    weight_mlp_cal_v = False
    apoz_mlp_cal_v = False
    sens_mlp_v = False
    weigh_prune_v = False
    # Setup CUDA, GPU & distributed training
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
            
        n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (local_rank, device, n_gpu, bool(local_rank != -1), fp16))
    

    set_seed(seed, n_gpu)
    print('using_gpu{}'.format(n_gpu))
    # Model & Tokenizer Setup
    model_type, img_size, pretrained_dir, device, model = setup(model_type, img_size, pretrained_dir, device, dataset)
    # print(model.named_children)
    loss = loss_P
    _, val_loader, _= cifar10.get_dataset_and_loaders(val_split=500, val_batch_size=64)
    model_back = model
    if mlp_prune == True:
        model=mlp_prune_f(model)
        # output_dir = "/workspace/CICC-main/experiments train - using/main_net/output_cifar10_formal1_prunemlp"
    if head_prune == True:
        head_prune_f(model)
        # output_dir = "/workspace/CICC-main/experiments train - using/main_net/output_cifar10_formal"
        os.makedirs(output_dir, exist_ok=True)
        save_model(output_dir,name,model)
    if test_module == True:
        test_module_f(model)
    if shapley_calculate_head == True:
        # loss = loss_P
        # _, val_loader, _= cifar10.get_dataset_and_loaders(val_split=500, val_batch_size=256)
        shapley_calculate_head_f(model,loss,val_loader,device)

    if shapley_calculate == True:
        res = mlp_layer_prune(model_back,loss,val_loader,device)
        name = 'shapley_512_448_5'
        model=any_mlp_prune(model_back)
        train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    if talor_mlp_prune_v == True:
        res = talor_mlp_cal(model_back,loss,val_loader,device)
        name = 'taylor_512_448'
        model=any_mlp_prune(model_back,res)
        train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    if weight_mlp_cal_v == True:
        res = weight_mlp_cal(model_back,loss,val_loader,device)
        name = 'weight_512_448'
        model=any_mlp_prune(model_back,res)
        train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    if apoz_mlp_cal_v == True:
        res = apoz_mlp_cal(model_back,loss,val_loader,device)
        name = 'apoz_512_448'
        model=any_mlp_prune(model_back,res)
        train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    if sens_mlp_v == True:
        res = sens_mlp_cal(model_back,loss,val_loader,device)
        name = 'sens_512_448'
        model=any_mlp_prune(model_back,res)
    train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    # sens_head_cal(model,loss,val_loader,device)
    # weight_head_cal(model,loss,val_loader,device)
    # talor_head_cal(model,loss,val_loader,device)
    # apoz_head_cal(model,loss,val_loader,device)
    # any_head_prune(model)
    print(model)
    # Training
    # train(local_rank, output_dir, name, train_batch_size, eval_batch_size, seed, n_gpu, gradient_accumulation_steps, dataset, img_size, learning_rate, weight_decay, num_steps, decay_type, warmup_steps, fp16, fp16_opt_level, device, max_grad_norm, eval_every, model)
    # sens_mlp_cal(model,loss,val_loader,device)
    print(123)

if __name__ == '__main__':
    main()
    





##模型可视化做不出来
    # 针对有网络模型，但还没有训练保存 .pth 文件的情况
    # import onnx
    # import onnx.utils
    # import onnx.version_converter
    # data = torch.randn(1, 3, 224, 224).cuda()
    # torch.onnx.export(
    #     model,
    #     data,
    #     '/workspace/CICC-main/experiments train - using/log/model.onnx',
    #     export_params=True,
    #     opset_version=8,
    # )

    # # 增加维度信息
    # model_file = '/workspace/CICC-main/experiments train - using/log/model.onnx'
    # onnx_model = onnx.load(model_file)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
    
    # x = torch.randn(1, 3, 224, 224).cuda()  # 随机生成一个输入
    # modelData = "/workspace/CICC-main/experiments train - using/log/demo.pth"  # 定义模型数据保存的路径
    # # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
    # torch.onnx.export(model, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构

    #  针对已经存在网络模型 .pth 文件的情况
    # import netron

    # modelData = "./demo.pth"  # 定义模型数据保存的路径
    # netron.start(modelData)  # 输出网络结构

    # writer = SummaryWriter(comment='test_your_comment',filename_suffix="_test_your_filename_suffix")
    # images = torch.zeros(1, 3, 224, 224).to(device).cuda()#要求大小与输入图片的大小一致
    # writer.add_graph(model, images, verbose=False)    
    # writer.close()
    # from torchsummary import summary
    # print(summary(model, (1,3,32,32), device="cuda"))
