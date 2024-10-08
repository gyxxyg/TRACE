# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower, build_time_tower, build_score_tower, build_sync_tower
from .multimodal_projector.builder import build_vision_projector
from ..mm_utils import get_anyres_image_grid_shape
from ..constants import NUM_FRAMES, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_MMODAL_PATCH_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX, MMODAL_INDEX_TOKEN


class TraceMetaModel:

    def __init__(self, config):
        super(TraceMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)
        
        self.time_tokenizer, self.time_tower = build_time_tower(None, None, 4096)
        self.score_tokenizer, self.score_tower = build_score_tower(None, None, 4096)
        self.sync_tower = build_sync_tower(4096)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_time_tower(self):
        time_tower = getattr(self, 'time_tower', None)
        return time_tower

    def get_score_tower(self):
        score_tower = getattr(self, 'score_tower', None)
        return score_tower

    def get_sync_tower(self):
        sync_tower = getattr(self, 'sync_tower', None)
        return sync_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        downsample_num = model_args.downsample_num

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.downsample_num = downsample_num

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


    ##################################################################################

    def initialize_time_modules(self, model_args, pretrained_tokenizer=None, pretrained_embedding_weights=None, dim=4096):

        # pretrained_embedding = self.embed_tokens.weight.data
        # print(pretrained_embedding)

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if self.get_time_tower() is None:
            print('initialize time tower from scarch')
            self.time_tokenizer, self.time_tower = build_time_tower(pretrained_tokenizer, pretrained_embedding_weights=pretrained_embedding_weights, dim=dim)
        else:
            print('use existing time tower')
            return

        if pretrain_mm_mlp_adapter is not None:
            print('load time tower weights from checkpoint')
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.time_tower.load_state_dict(get_w(mm_projector_weights, 'time_tower'), strict=True)


    def initialize_score_modules(self, model_args, tokenizer=None, pretrained_embedding_weights=None, dim=4096):

        # pretrained_embedding = self.embed_tokens.weight.data
        # print(pretrained_embedding_weights)

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if self.get_score_tower() is None:
            print('initialize score tower from scarch')
            self.score_tokenizer, self.score_tower = build_score_tower(tokenizer, pretrained_embedding_weights=pretrained_embedding_weights, dim=dim)
        else:
            print('use existing score tower')
            return

        if pretrain_mm_mlp_adapter is not None:
            print('load score tower weights from checkpoint')
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            #self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.score_tower.load_state_dict(get_w(mm_projector_weights, 'score_tower'), strict=True)

    ##################################################################################





class TraceMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_time_tower(self):
        return self.get_model().get_time_tower()

    def get_score_tower(self):
        return self.get_model().get_score_tower()

    def get_sync_tower(self):
        return self.get_model().get_sync_tower()

    def encode_images_or_videos(self, images_or_videos, modalities, video_timestamps, seperate_time_feature=True):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        videos = [x.unsqueeze(0).expand(num_frames, -1, -1, -1) if modal == 'image' else x for x, modal in zip(images_or_videos, modalities)]
        videos = torch.stack(videos, dim=0)


        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
        hw = int(frames_features.shape[2] ** 0.5)

        # v5
        if seperate_time_feature:
            frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw)) # b t s d
        ##################################################################################
        if video_timestamps is not None: # b t
            video_time_tokens = self.encode_time(video_timestamps) # b t n_t
            time_features = []
            for batch_idx in range(batch_size):
                batch_time_features = []
                for f_idx in range(len(video_time_tokens[batch_idx])): # [time_1_tokens, time_2_tokens, ..., time_t_tokens]
                    cur_time_features = self.get_time_tower()(video_time_tokens[batch_idx][f_idx][:-1].to(frames_features.device)) # do not use the <sync> token here
                    # print(f'{cur_time_features.shape} {video_time_tokens[batch_idx][f_idx]} {video_timestamps[batch_idx][f_idx]}')
                    batch_time_features.append(cur_time_features.unsqueeze(0)) # [(1, n_t, h), ...]
                batch_time_features = torch.cat(batch_time_features, dim=0) # t, n_t, h
                time_features.append(batch_time_features.unsqueeze(0))
            time_features = torch.cat(time_features, dim=0).to(frames_features.device) # b, t, n_t, h
            
            if not seperate_time_feature:
                time_features = time_features.view(time_features.shape[0], time_features.shape[1], -1, frames_features.shape[-1]) # remove it if v5

                frames_features = torch.cat([frames_features, time_features], dim=2) # b t (s + n_t) h
                frames_features = self.temporal_aggregator(frames_features, hw, int(frames_features.shape[2] / hw))
            else:
                frames_features = torch.cat([frames_features, time_features], dim=2) # b t (s + n_t) h
                # print(frames_features.shape, time_features.shape)
            frames_features =  einops.rearrange(frames_features, 'b t n h -> b (t n) h')

            
            # v5
            # frames_features =  einops.rearrange(frames_features, 'b t n h -> b (t n) h')
            ##################################################################################
            

        return frames_features


    ##################################################################################

    def encode_time(self, times): 

        ## [
        #   [[t_1, t_2], [t_3, t_4], [t_5, t_6], ... ],
        #   [[t_1], [t_2], [t_3]],
        #   [[t_1, t_2]] 
        #  ]

        # if not self.get_model().get_time_tower().initialized:
        #     self.get_model().get_time_tower().update_weights(self.get_model().embed_tokens.weight)

        time_tokens = []
        for batch_times in times: # for each batch
            batch_time_tokens = [self.get_model().get_time_tower().encode(t) for t in batch_times]
            assert all([batch_time_token.shape == batch_time_tokens[0].shape for batch_time_token in batch_time_tokens]), f'{batch_times} {[batch_time_token.shape for batch_time_token in batch_time_tokens]}'
            time_tokens.append(batch_time_tokens)

        return time_tokens # [[event1-tokens, event2-tokens], ..., [event1-tokens, event2-tokens]]

    def encode_score(self, scores): 

        ## [
        #   [[s1], [s2], [s3]],
        #   [[], [], []],
        #   [[s1], [s2]]
        #  ]

        # if not self.get_model().get_score_tower().initialized:
        #     self.get_model().get_score_tower().update_weights(self.get_model().embed_tokens.weight)

        score_tokens = []
        for batch_scores in scores: # for each batch
            batch_score_tokens = [self.get_model().get_score_tower().encode(s) for s in batch_scores]
            score_tokens.append(batch_score_tokens)

        return score_tokens # [[event1-tokens, event2-tokens], ..., [event1-tokens, event2-tokens]]


    ##################################################################################

    def temporal_aggregator(self, frames_features, h, w):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features, h, w)
        elif "spatial_time_slot" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features, h * h)
        elif "slot" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")

        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities, times, scores, video_timestamps=None
    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            new_input_embeds = []
            for batch_idx in range(input_ids.shape[0]):
                cur_input_ids = input_ids[batch_idx]

                # embed text input ids
                cur_text_ids = cur_input_ids % self.vocab_size
                text_embeds = self.get_model().embed_tokens(cur_text_ids)
                # embed sync
                sync_positions = (cur_input_ids == self.vocab_size)
                sync_embeds = self.get_sync_tower()(cur_input_ids[sync_positions])
                # embed time input ids
                time_positions = cur_input_ids >= (self.vocab_size + 1) and cur_input_ids < (self.vocab_size + self.time_vocab_size + 1)
                cur_time_ids = cur_input_ids[time_positions] - self.vocab_size - 1
                time_embeds = self.get_time_tower()(cur_time_ids)
                # embed score input ids
                score_positions = cur_input_ids >= (self.vocab_size + self.time_vocab_size + 1)
                cur_score_ids = cur_input_ids[score_positions] - self.vocab_size - self.time_vocab_size - 1
                score_embeds = self.get_score_tower()(cur_score_ids)

                # combine all the things
                text_embeds[time_positions] = time_embeds
                text_embeds[score_positions] = score_embeds
                text_embeds[sync_positions] = sync_embeds

                new_input_embeds.append(text_embeds)
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

            return None, attention_mask, past_key_values, new_input_embeds, labels, None, None

        Xs, keys = X_modalities
        X_features = self.encode_images_or_videos(Xs, keys, video_timestamps)
        
        ##################################################################################
        time_tokens = self.encode_time(times)
        score_tokens = self.encode_score(scores)
        # print(time_tokens, torch.where(input_ids == MMODAL_TOKEN_INDEX['TIME']))

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_time_labels = [] if labels is not None else None
        new_score_labels = [] if labels is not None else None
        cur_X_idx = 0

        # print(f'{time_tokens}')

        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # cur_X_features = X_features[batch_idx]

            
            cur_time_tokens = torch.cat(time_tokens[batch_idx], dim=0) if len(time_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            cur_score_tokens =torch.cat(score_tokens[batch_idx], dim=0) if len(score_tokens[batch_idx]) > 0 else torch.tensor([], dtype=torch.int)
            # print(f'{cur_score_tokens} {cur_time_tokens}')
            cur_X_features = X_features[batch_idx]
            # cur_X_features = self.encode_images_or_videos([Xs[batch_idx]], [keys[batch_idx]], [video_timestamps[batch_idx]]).squeeze(0)
            # print(f'{Xs[batch_idx].shape} {cur_X_features.shape} {cur_X_features.device}')
            cur_time_features = self.get_time_tower()(cur_time_tokens.to(cur_X_features.device))
            cur_score_features = self.get_score_tower()(cur_score_tokens.to(cur_X_features.device))
            # print(cur_score_features.shape)

            # print('finish score')

            video_position = torch.where((cur_input_ids == MMODAL_TOKEN_INDEX['VIDEO']) + (cur_input_ids == MMODAL_TOKEN_INDEX['IMAGE']))[0]
            assert len(video_position) == 1, "only have one video inputs!"
            video_position = video_position[0]

            cur_new_input_ids = torch.cat([cur_input_ids[:video_position], torch.full((cur_X_features.shape[0],), MMODAL_TOKEN_INDEX['VIDEO'], device=cur_input_ids.device, dtype=cur_input_ids.dtype), cur_input_ids[video_position+1:]], dim=0)
            cur_sync_features = self.get_sync_tower()(cur_new_input_ids[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']].to(cur_X_features.device))

            cur_text_input_ids = torch.clamp(cur_new_input_ids, min=0)
            cur_new_input_embeds = self.get_model().embed_tokens(cur_text_input_ids)

            # print(f"{torch.sum(cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME'])} {cur_time_features.shape} {times[batch_idx]}")
            # print(cur_sync_features.shape)

            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['VIDEO']] = cur_X_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_features
            cur_new_input_embeds[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = cur_sync_features

            # print(f'{cur_new_input_embeds.shape} {cur_new_input_embeds.device}')

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = torch.cat([cur_labels[:video_position], torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype), cur_labels[video_position+1:]], dim=0)

                cur_new_labels[cur_new_input_ids < 0] = IGNORE_INDEX
                cur_new_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SYNC']] = self.vocab_size

                # print(torch.where(cur_new_labels != IGNORE_INDEX), cur_new_labels[cur_new_labels != IGNORE_INDEX])

                cur_time_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)

                # print(cur_time_tokens.shape, torch.sum(cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']))

                cur_time_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['TIME']] = cur_time_tokens.to(cur_labels.device)

                cur_score_labels = torch.full(cur_new_labels.shape, IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)

                cur_score_labels[cur_new_input_ids == MMODAL_TOKEN_INDEX['SCORE']] = cur_score_tokens.to(cur_labels.device)

            
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                new_labels.append(cur_new_labels)
                new_time_labels.append(cur_time_labels)
                new_score_labels.append(cur_score_labels)

        # assert 1 < 0
        ##################################################################################
        # padding
        # max_len = 2048
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
                
                ##################################################################################

                new_time_labels_align = []
                for cur_new_time_label in new_time_labels:
                    cur_new_time_label = torch.cat((cur_new_time_label, torch.full((max_len - cur_new_time_label.shape[0],), IGNORE_INDEX, dtype=cur_new_time_label.dtype, device=cur_new_time_label.device)), dim=0)
                    new_time_labels_align.append(cur_new_time_label)
                new_time_labels = torch.stack(new_time_labels_align, dim=0)

                new_score_labels_align = []
                for cur_new_score_label in new_score_labels:
                    cur_new_score_label = torch.cat((cur_new_score_label, torch.full((max_len - cur_new_score_label.shape[0],), IGNORE_INDEX, dtype=cur_new_score_label.dtype, device=cur_new_score_label.device)), dim=0)
                    new_score_labels_align.append(cur_new_score_label)
                new_score_labels = torch.stack(new_score_labels_align, dim=0)

                ##################################################################################

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

                ##################################################################################

                new_time_labels = torch.stack(new_time_labels, dim=0)
                new_score_labels = torch.stack(new_score_labels, dim=0)

                ##################################################################################

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # print(f'{new_input_embeds.shape}, {new_labels.shape}, {new_time_labels.shape}, {new_score_labels.shape}')
        # print(f'{new_labels.device}, {torch.where(new_labels != -100)}, {torch.where(new_time_labels != -100)}, {torch.where(new_score_labels != -100)}')
        # print(new_labels[new_labels != -100])
        # print(f'prepare end {new_labels.device}')

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_time_labels, new_score_labels
        # return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings  = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg  = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:]  = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def initialize_MM_tokenizer(self, model_args, tokenizer):
        ##################################################################################
        if model_args.mm_use_im_patch_token:
            for modal in ['IMAGE', 'VIDEO', 'AUDIO', 'TIME', 'SCORE']:
                tokenizer.add_tokens([DEFAULT_MMODAL_PATCH_TOKEN[modal.upper()]], special_tokens=True)
            # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = 0
            for modal in ['IMAGE', 'VIDEO', 'AUDIO', 'TIME', 'SCORE']:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_MMODAL_START_TOKEN[modal.upper()], DEFAULT_MMODAL_END_TOKEN[modal.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))


            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 6  # start/end tokens for image/video/audio
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
        ##################################################################################