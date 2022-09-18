# reference : https://github.com/andrebola/contrastive-mir-learning

import jax
import jax.numpy as jnp
from jax import random
import optax

from flax import linen as nn
from typing import Callable, Any, Optional


class Conv_2d(nn.Module):
    
    output_channel:int
    shape:int=3
    stride:int=1
    pooling:int=2
    train:bool = True
    def setup(self):
        self.conv = nn.Conv(self.output_channel, kernel_size=(self.shape, self.shape), strides=self.stride)
        self.bn = nn.normalization.BatchNorm(use_running_average=self.train)
        
    def __call__(self, x):
        x = self.conv(x)
        x = jax.nn.relu(self.bn(x))
        x = nn.max_pool(x, window_shape=(self.pooling), strides=(self.pooling))
        return x 
    
class fc_audio(nn.Module):
    size_w_rep: int = 512
    train: bool = True    
    @nn.compact
    def __call__(self, x):
        x = jax.nn.relu(nn.Dense(512)(x))
        x = nn.Dropout(rate=0.5)(x, deterministic=not self.train)
        x = nn.Dense(self.size_w_rep)(x)
        x = nn.LayerNorm()(x)
        
        return x   
    
class AudioEncoder(nn.Module):
    
    size_w_rep: int = 512    
    train: bool = True    
    temperature : float = 0.07
    
    def setup(self):
        self.audio_encoder = nn.Sequential([
            nn.normalization.BatchNorm(use_running_average=self.train),
            Conv_2d(128, pooling=(2,2), train=self.train), # 24 X 938
            Conv_2d(128, pooling=(2,2), train=self.train), # 12 X 469
            Conv_2d(256, pooling=(2,2), train=self.train), # 6 X 234
            Conv_2d(256, pooling=(2,4), train=self.train), # 3 X 58
            Conv_2d(256, pooling=(1,3), train=self.train), # 3 X 19
            Conv_2d(256, pooling=(1,3), train=self.train), # 3 X 6
            Conv_2d(512, pooling=(2,3), train=self.train)  # 1 X 2          
            ])
        
        self.fc_audio = fc_audio(self.size_w_rep, self.train)

    def __call__(self, x):
        x = self.audio_encoder(x)
        x = x.reshape(x.shape[0], -1)
        feats = self.fc_audio(x)
        
#         cos_sim = optax.cosine_similarity(feats[:,None,:], feats[None,:,:])
#         cos_sim /= self.temperature
#         diag_range = jnp.arange(feats.shape[0], dtype=jnp.int32)
#         cos_sim = cos_sim.at[diag_range, diag_range].set(-9e15)
        
#         shifted_diag = jnp.roll(diag_range, x.shape[0]//2)
#         pos_logits = cos_sim[diag_range, shifted_diag]
        
#         # InfoNCE loss
#         nll = - pos_logits + nn.logsumexp(cos_sim, axis=-1)
#         nll = nll.mean()

#         # Logging
#         metrics = {'loss': nll}
#         # Determine ranking position of positive example
#         comb_sim = jnp.concatenate([pos_logits[:,None],
#                                     cos_sim.at[shifted_diag, diag_range].set(-9e15)],
#                                    axis=-1)
#         sim_argsort = (-comb_sim).argsort(axis=-1).argmin(axis=-1)
        
#         # Logging of ranking position
#         metrics['acc_top1'] = (sim_argsort == 0).mean()
#         metrics['acc_top5'] = (sim_argsort < 5).mean()
#         metrics['acc_mean_pos'] = (sim_argsort + 1).mean()

        return feats

    def encode(self, x):
        x = self.audio_encoder(x)
        return x 
