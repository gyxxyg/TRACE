#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "stc_connector":
        return STCConnector(config)
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    elif projector_type == "slot":
        return SlotPool(config)
    elif projector_type == "spatial_slot":
        return SpatialSlotPool(config)
    elif projector_type == 'spatial_time_slot':
        return SpatialTimeSlotPool(config)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        # self.downsample = (downsample[0], downsample[1] * downsample[2])
        self.downsample_num = config.downsample_num
        print(f'downsample num {self.downsample_num}')
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True
            ),
            nn.SiLU()
        )
        # self.sampler = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=hidden_size,
        #         out_channels=hidden_size,
        #         kernel_size=self.downsample,
        #         stride=self.downsample,
        #         padding=1,
        #         bias=True
        #     ),
        #     nn.SiLU()
        # )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x, h=24, w=24):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=h, w=w)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")

        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        
        
        # # 1. the first stage of the adapter
        # x = self.s1(x)
        # x = einops.rearrange(x, "(b t) d h w -> b d t (h w)", t=t)

        # for _ in range(self.downsample_num):
        #     x = self.sampler(x)
        # new_t = x.size(2)
        # # 3. the second stage of the adapter
        # x = x.unsqueeze(-1)
        # x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # x = self.s2(x)
        # x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        # x = self.readout(x)
        return x


class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU())


class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


# copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
# TODO @Arthur no longer copied from LLama after static cache
class SlotRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# TODO @Arthur no longer copied from LLama after static cache
def apply_rotary_pos_emb(x, cos, sin, position_ids):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids]
    sin = sin[position_ids]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class SlotPool(nn.Module):

    def __init__(self, config, num_slots=1024):
        super().__init__()

        self.slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = SlotRotaryEmbedding(
            config.mm_hidden_size
        )
        self.num_slots = num_slots


    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        
        t = x.size(1)

        # if x.ndim == 5:
        #     x = einops.rearrange(x, "b t h w d -> b t (h w) d") # b t n d

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> b (t n) d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> b (t h w) d") # b n d

        x = self.ln_vision(x)

        position_ids = torch.repeat_interleave(torch.tensor(range(t)), repeats=n).to(x.device)
        cos, sin = self.rotary_emb(x, seq_len=t)
        x = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        logits = torch.matmul(x, self.slots) # b n s
        logits = torch.softmax(logits, dim=1) # b n s

        res = torch.matmul(x.permute(0,2,1), logits).permute(0, 2, 1) # b s d

        return self.readout(res)



class SpatialSlotPool(nn.Module):

    def __init__(self, config, num_slots=8):
        super().__init__()

        print(num_slots)

        self.slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = SlotRotaryEmbedding(
            config.mm_hidden_size
        )
        self.num_slots = num_slots


    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        
        t = x.size(1)

        # if x.ndim == 5:
        #     x = einops.rearrange(x, "b t h w d -> b t (h w) d") # b t n d

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> (b t) n d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> (b t) (h w) d") # (b t) n d

        # print(self.ln_vision.weight)
        x = self.ln_vision(x)

        position_ids = torch.arange(
                n, dtype=torch.long, device=x.device
            )
        cos, sin = self.rotary_emb(x, seq_len=n)
        x = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        logits = torch.matmul(x, self.slots) # (b t) n s
        logits = torch.softmax(logits, dim=1) # (b t) n s
        # print(logits[0, :, -1])
        # print(logits.shape)

        res = torch.matmul(x.permute(0,2,1), logits).permute(0, 2, 1) # (b t) s d
        # res = einops.rearrange(res, "(b t) s d -> b (t s) d", t=t) # (b t) n d
        # v5
        res = einops.rearrange(res, "(b t) s d -> b t s d", t=t) # (b t) n d

        return self.readout(res)

class SpatialTimeSlotPool(nn.Module):

    def __init__(self, config, num_spatial_slots=8, num_time_slots=1):
        super().__init__()


        self.spatial_slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_spatial_slots))
        self.time_slots = nn.Parameter(torch.randn(config.mm_hidden_size, num_time_slots))
        self.ln_vision = LayerNorm(config.mm_hidden_size)
        self.readout = nn.Linear(config.mm_hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = SlotRotaryEmbedding(
            config.mm_hidden_size
        )
        self.num_spatial_slots = num_spatial_slots
        self.num_time_slots = num_time_slots

    
    def forward(self, x, image_dim=576):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        
        
        t = x.size(1)

        if x.ndim == 4:
            n = x.size(2)
            x = einops.rearrange(x, "b t n d -> (b t) n d")
        elif x.ndim == 5:
            n = x.size(2) * x.size(3)
            x = einops.rearrange(x, "b t h w d -> (b t) (h w) d") # (b t) n d

        # for image part
        
        image_x, time_x = torch.split(x, image_dim, dim=1)

        print(image_x.shape, time_x.shape)

        image_x = self.ln_vision(image_x)

        image_position_ids = torch.arange(
                image_dim, dtype=torch.long, device=image_x.device
            )
        image_cos, image_sin = self.rotary_emb(image_x, seq_len=image_dim)
        image_x = apply_rotary_pos_emb(image_x, image_cos, image_sin, image_position_ids)
        
        image_logits = torch.matmul(image_x, self.spatial_slots) # (b t) n s
        image_logits = torch.softmax(image_logits, dim=1) # (b t) n s
        # print(logits.shape)

        image_outputs = torch.matmul(image_x.permute(0,2,1), image_logits).permute(0, 2, 1) # (b t) s d
        image_outputs = einops.rearrange(image_outputs, "(b t) s d -> b t s d", t=t) # (b t) n d
        image_outputs = self.readout(image_outputs)

        print(image_outputs.shape)

        # for time part
        time_position_ids = torch.arange(
                n - image_dim, dtype=torch.long, device=time_x.device
            )
        time_cos, time_sin = self.rotary_emb(time_x, seq_len=n - image_dim)
        time_x = apply_rotary_pos_emb(time_x, time_cos, time_sin, time_position_ids)

        time_logits = torch.matmul(time_x, self.time_slots) # (b t) n s
        time_logits = torch.softmax(time_logits, dim=1) # (b t) n s
        # print(logits.shape)

        time_outputs = torch.matmul(time_x.permute(0,2,1), time_logits).permute(0, 2, 1) # (b t) s d
        time_outputs = einops.rearrange(time_outputs, "(b t) s d -> b t s d", t=t) # (b t) n d

        print(time_outputs.shape)

        # for final outputs
        outputs = torch.cat([image_outputs, time_outputs], dim=2)
        # outputs = einops.rearrange(outputs, "b t s d -> b t s) d")


        return outputs