# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the Licmmense at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import copy
import json
import random
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

import cv2
import decord
import imageio
import traceback
import numpy as np
import transformers
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

sys.path.append('yourpath/projects')
sys.path.append('yourpath/projects/Trace')
sys.path.append('yourpath/projects/Trace/trace')
from Trace.trace import conversation as conversation_lib
from Trace.trace.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_INDEX_TOKEN
from Trace.trace.trace_trainer import TraceTrainer
from Trace.trace.model import *
from Trace.trace.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, tokenizer_MMODAL_token_all

os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_embed_head: bool = field(default=False)
    tune_lm_embed_head: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Other Arguments
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    downsample_num: int = field(default=1)


@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    sample_scheme: Optional[str] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'score', 'time', 'sync']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head') 
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_mm_mlp_adapter", False) or getattr(trainer.args, 'tune_mm_embed_head', False):
    #     # Only save Adapter
    #     keys_to_match = []
    #     if getattr(trainer.args, "tune_mm_mlp_adapter", False):
    #         keys_to_match = ['mm_projector']
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(['embed_tokens', 'embed_in'])
    #     ##################################################################################################
    #     if getattr(trainer.args, 'tune_mm_embed_head', False):
    #         keys_to_match.extend(['embed_tokens', 'head'])
    #     ###################################################################################################

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split('/')[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith('checkpoint-'):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     return

    if trainer.deepspeed:
        rank0_print('Saving...')
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data


        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # NOTE: scan token of each modal and move them to the beginning of the sentence. 
            for DEFAULT_TOKEN in DEFAULT_MMODAL_TOKEN.values():
                MODAL_TYPE = None
                if DEFAULT_TOKEN in sentence['value'] and 'time' not in DEFAULT_TOKEN and 'score' not in DEFAULT_TOKEN and 'sync' not in DEFAULT_TOKEN:
                    MODAL_TYPE = DEFAULT_TOKEN[1:-1]
                    sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, f'<{MODAL_TYPE.capitalize()}>' + DEFAULT_TOKEN + f'</{MODAL_TYPE.capitalize()}>')
                replace_token = DEFAULT_TOKEN
                if data_args.mm_use_im_start_end and MODAL_TYPE is not None:
                    replace_token = DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()] + replace_token + DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()]
                sentence["value"] = sentence["value"].replace(DEFAULT_TOKEN, replace_token)

    return sources

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 1. Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 2. Tokenize conversations
    if len(MODAL_list) > 0:
        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN

    # 3. Prepare training inputs and labels.
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        cur_len = 0
        rounds = conversation.split(conv.sep)
        # 3.1 Ignore system prompt (zero order round)
        round_len = len(tokenizer(rounds[0]).input_ids) + 1
        target[cur_len:cur_len+round_len] = IGNORE_INDEX
        cur_len += round_len
        rounds = rounds[1:]

        # QA rounds
        for i, rou in enumerate(rounds):
            if rou == "" or rou == '\n':
                break

            role = conv.roles[i % 2]
            parts = rou.split(role)

            assert len(parts) == 2, f"Invalid conversation: {rou}"
            parts[0] += role

            if len(MODAL_list) > 0:
                round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) + 1
                instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i % 2 == 0:
                # 3.2 Ignore role & instruction
                target[cur_len:cur_len+round_len] = IGNORE_INDEX
            else:
                # 3.3 Ignore role & train response
                target[cur_len:cur_len+instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

    # TODO: Fixing this hardcoding for qwen/ChatML template
    if "qwen" in conv.version:
        for input_id, target in zip(input_ids, targets):
            # <|im_start|>, <|im_end|>
            target[input_id == 151644] = 151644
            target[input_id == 151645] = 151645

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print(conversations)

    # Tokenize conversations
    if len(MODAL_list) > 0:
        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids = torch.stack([tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                round_len = len(tokenizer_MMODAL_token_all(rou, tokenizer))
                instruction_len = len(tokenizer_MMODAL_token_all(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # print(input_ids)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    assert len(sources) == len(MODAL_list)
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        # source is the conversations in the input data
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print(conversation_lib.default_conversation.sep)
    print(conversations)

    # Tokenize conversations
    if len(MODAL_list) > 0:
        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids = torch.stack([tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    #for conversation, target in zip(conversations, targets):
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer)) 
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                # fix the issue of tokenization mismatch
                round_len = len(tokenizer_MMODAL_token_all(rou, tokenizer))
                instruction_len = len(tokenizer_MMODAL_token_all(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list=[]
) -> Dict:
    # add end signal and concatenate together
    # print(sources)
    conversations = []
    DEFAULT_TOKEN = DEFAULT_MMODAL_TOKEN[MODAL_list[0]]
    for source in sources:
        assert len(source) == 2
        source[0]['value'] = DEFAULT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # print(conversations)
    # tokenize conversations
    input_ids = [tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_MMODAL_token_all(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    
    # print(input_ids, targets, sources)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list: list = []
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, MODAL_list)
    # qwen2 conversation style preprocess
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN:
        return preprocess_qwen(sources, tokenizer, MODAL_list)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts, token_index):
        return [len(tokenizer_MMODAL_token_all(prompt, tokenizer)) for prompt in prompts]

    if len(MODAL_list) > 0:
        input_ids = [tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt') for i, prompt in enumerate(conversations)]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for idx, (target, source) in enumerate(zip(targets, sources)):
        if len(MODAL_list) > 0:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source], MODAL_list[idx])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 513 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # cur_len = sum(sum(conv['value'].count(k) for k in MMODAL_INDEX_TOKEN.values()) for conv in sample['conversations'])
            # length_list.append(cur_len)

            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames
        sample_scheme = 'uniform' if self.data_args.sample_scheme is None else self.data_args.sample_scheme
        # print(num_frames)

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        ##############################################################################################################

        times = []
        scores = []
        times_count = 0
        scores_count = 0
        for e in sources:
            times.extend(e['times'])
            scores.extend(e['scores'])

        ##############################################################################################################
        MODAL_list = []
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_file = os.path.join(self.data_args.data_folder, image_file)

            try:
                image = process_image(image_file, image_processor, self.data_args.image_aspect_ratio)[0]
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict)-1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            MODAL_list.append('IMAGE')
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_file = os.path.join(self.data_args.data_folder, video_file)

            try: 
                video, video_timestamps = process_video(video_file, video_processor, self.data_args.image_aspect_ratio, num_frames, sample_scheme=sample_scheme)
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict)-1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            MODAL_list.append('VIDEO')
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # NOTE: for sharegpt data in the sft stage, we use the default IMAGE as modal token
            MODAL_list.append('IMAGE')

        data_dict = preprocess(sources, self.tokenizer, MODAL_list=MODAL_list)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        ##############################################################################################################

        data_dict['time'] = times
        data_dict['score'] = scores

        # print(f'{video_file} {times}')

        ##############################################################################################################

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['video_timestamps'] =[[0]] * num_frames
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
            data_dict['video_timestamps'] = video_timestamps
            # print('*' * 100)
            # print(data_dict['video_timestamps'])
            # print(data_dict['time'])
            data_dict['time'] = [[min(data_dict['video_timestamps'], key=lambda x: abs(x[0] - target))[0] for target in interval] for interval in data_dict['time']]
            # print(data_dict['time'])
            # print('*' * 100)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        # print(data_dict)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        Xs, keys = [], []
        ##############################################################################################################
        times, scores, video_timestamps = [], [], []
        for instance in instances:
            for x in DEFAULT_MMODAL_TOKEN.keys():
                x = x.lower()
                if x in instance:
                    if 'time' in x:
                        times.append(instance[x])
                    elif 'score' in x:
                        scores.append(instance[x])
                    elif 'sync' in x:
                        continue
                    else:
                        Xs.append(instance[x])
                        keys.append(x)
            video_timestamps.append(instance['video_timestamps'])

        batch['images'] = [Xs, keys]  # we do not change the key's name.
        batch['times'] = times
        batch['scores'] = scores
        batch['video_timestamps'] = video_timestamps
        return batch
        ##############################################################################################################


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    # rank0_print(f'Dataset: {train_dataset[0]}')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'vicuna' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            config.downsample_num = model_args.downsample_num
            model = TraceLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'mixtral' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            config.downsample_num = model_args.downsample_num
            model = TraceMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif 'qwen2' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            config.downsample_num = model_args.downsample_num
            model = TraceQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        else:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            config.downsample_num = model_args.downsample_num
            model = TraceMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                # ignore_mismatched_sizes=True,
                **bnb_model_from_pretrained_args
            )
    else:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        config.downsample_num = model_args.downsample_num
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        print('Freezed Backbone')
        model.model.requires_grad_(False)
    else:
        model.model.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # print(model.get_input_embeddings(), model.get_input_embeddings().weight)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None: 
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            if model_args.version == "v1":
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            elif model_args.version == "v1_mistral":
                conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]

    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        ##################################################################################################################
        model.config.tune_mm_embed_head = training_args.tune_mm_embed_head = model_args.tune_mm_embed_head
        model.config.tune_lm_embed_head = training_args.tune_lm_embed_head = model_args.tune_lm_embed_head
        ##################################################################################################################
        if model_args.tune_mm_mlp_adapter:
            if model_args.freeze_backbone:
                model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        # else:
        #      model.get_model().mm_projector.to(device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_MM_tokenizer(model_args, tokenizer=tokenizer)

        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    # ##############################################################################################################

    # rank0_print("Current model:", model)

    # #initialize time and score towers
    rank0_print('Initializing time and score towers')
    model.get_model().initialize_time_modules(model_args=model_args)
    model.get_model().initialize_score_modules(model_args=model_args)
    # do not add new token in v2
    # model.initialize_embed_and_heads(model_args=model_args, tokenizer=tokenizer)

    # ##############################################################################################################

    for n, p in model.named_parameters():
        if getattr(training_args, 'tune_mm_embed_head', False):
            if 'time' in n or 'score' in n or 'sync' in n:
                p.requires_grad = True
        else:
            if 'time' in n or 'score' in n or 'sync' in n:
                p.requires_grad = False
        if getattr(training_args, 'tune_lm_embed_head', False):
            if ('head' in n or 'embed_tokens' in n) and ('time' not in n) and ('score' not in n) and ('sync' not in n):
                p.requires_grad = True
        else:
            if ('head' in n or 'embed_tokens' in n) and ('time' not in n) and ('score' not in n) and ('sync' not in n):
                p.requires_grad = False

        if 'lora' in n:
            p.requires_grad = True
        if p.requires_grad == True:
        # if not isinstance(p, torch.Tensor):
            rank0_print(n)
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    trainer = TraceTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        # state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        # if training_args.local_rank == 0 or training_args.local_rank == -1:
        #     model.config.save_pretrained(training_args.output_dir)
        #     model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        #     torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        # print('Merging LoRA weights...')
        torch.cuda.synchronize()
        trainer.model = trainer.model.merge_and_unload()
        print(trainer.model)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
