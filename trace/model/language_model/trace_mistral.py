# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Any
from torch import Tensor

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..trace_arch import TraceMetaModel, TraceMetaForCausalLM

class LMHeadBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, logits: Tensor):
        return logits

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        # print(grad_output.shape)
        grad_output[:, :, :-1] = 0
        # print(grad_output)
        return grad_output

class LMEmbedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, embeddings: Tensor):
        return embeddings

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        print(grad_output.shape)
        grad_output[:, :-1] = 0
        # print(grad_output)
        return grad_output

def lm_embed_hook(module, input, output):
    output = LMEmbedBackward.apply(output)

class TraceMistralConfig(MistralConfig):
    model_type = "trace_mistral"


class TraceMistralModel(TraceMetaModel, MistralModel):
    config_class = TraceMistralConfig

    def __init__(self, config: MistralConfig):
        super(TraceMistralModel, self).__init__(config)


class TraceMistralForCausalLM(MistralForCausalLM, TraceMetaForCausalLM):
    config_class = TraceMistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)

        # config.time_vocab_size = 12
        # config.score_vocab_size = 12

        self.model = TraceMistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.swap_tokens = {config.vocab_size: 1,
                            config.vocab_size + 1: 2, 
                            config.vocab_size + config.time_vocab_size + 1: 0}
        # #####################################################################################################

        self.time_head = nn.Linear(config.hidden_size, config.time_vocab_size, bias=False)

        self.score_head = nn.Linear(config.hidden_size, config.score_vocab_size, bias=False)
        self.sync_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.time_vocab_size = config.time_vocab_size
        self.score_vocab_size = config.score_vocab_size

        # hook_handle = self.get_model().embed_tokens.register_forward_hook(lm_embed_hook)
        # #####################################################################################################

        # Initialize weights and apply final processing
        self.post_init()

    def initialize_embed_and_heads(self, model_args, tokenizer):

        tokenizer.add_tokens(['<sync>'], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))



    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        times: Optional[List[List[List[float]]]] = None,
        scores: Optional[List[List[List[float]]]] = None,
        video_timestamps: Optional[List[List[List[float]]]] = None,
        heads: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print(input_ids)


        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                time_labels, 
                score_labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                times,
                scores,
                video_timestamps=video_timestamps
            )

        # print(f'prepared {labels[labels > 0]} {labels > 0}')
        # print(self.time_head.weight)

        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        sync_logits = self.sync_head(hidden_states)
        logits = torch.cat([logits, sync_logits], dim=-1)
        logits = logits.float()

        #####################################################################################################
        time_logits = self.time_head(hidden_states)
        time_logits = time_logits.float()
        score_logits = self.score_head(hidden_states)
        score_logits = score_logits.float()

        # logits = torch.cat([logits, time_logits, score_logits], dim=-1)

        logits_list = [logits, time_logits, score_logits]
        if labels is not None:
            labels_list = [labels, time_labels, score_labels]
        # logits_list = [time_logits]
        # labels_list = [time_labels]
        # vocab_size_list = [self.config.time_vocab_size]
        # modals = ['time']
        vocab_size_list = [self.vocab_size + 1, self.config.time_vocab_size, self.config.score_vocab_size]
        # modals = ['text', 'time', 'score']
        # for m, l in zip(modals, labels_list):
        #     print(f'{m}, {torch.max(l)}')
        # print(vocab_size_list)
        # print(logits_list)

        loss = None
        if labels is not None:
            loss = []
            loss_fct = CrossEntropyLoss()

            for cur_logits, cur_labels, cur_vocab_size in zip(logits_list, labels_list, vocab_size_list):

                shift_logits = cur_logits[..., :-1, :].contiguous()
                shift_labels = cur_labels[..., 1:].contiguous()
                # print(f'cur logits {shift_logits}, labels {shift_labels}, shapes, {shift_logits.shape}, {shift_labels.shape}, {cur_modal}')
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, cur_vocab_size)
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                loss.append(loss_fct(shift_logits, shift_labels))
            # print(loss)
                # print(f'{cur_labels[cur_labels != -100]} {cur_logits[cur_labels != 100]} {loss_fct(shift_logits, shift_labels)}')

            loss = sum(loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        if heads is not None:
            assert len(heads) == logits.shape[0]
            logits = torch.cat([logits, time_logits, score_logits], dim=-1)
            # print(logits.shape)
            vocab_size_list = [(0, self.vocab_size + 1), (self.vocab_size + 1, self.vocab_size + self.config.time_vocab_size + 1), (self.vocab_size + self.config.time_vocab_size + 1, self.vocab_size + self.config.time_vocab_size + self.config.score_vocab_size + 1)]
            for batch_idx, head in enumerate(heads):
                cur_vocab_size = vocab_size_list[head]
                logits[batch_idx, ..., :cur_vocab_size[0]] = float('-inf')
                logits[batch_idx, ..., cur_vocab_size[1]:] = float('-inf')
        
        # print(logits.shape, logits)
        # print(loss)
        

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        #####################################################################################################

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        times: Optional[List[List[List[float]]]] = None,
        scores: Optional[List[List[List[float]]]] = None,
        video_timestamps: Optional[List[List[List[float]]]] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # print(inputs)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images_or_videos is not None:
            if times is None:
                times = [[]] * len(images_or_videos)
            if scores is None:
                scores = [[]] * len(images_or_videos)
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _, _, _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                X_modalities=[images_or_videos, modal_list],
                times=times,
                scores=scores,
                video_timestamps=video_timestamps
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # print(input_ids)
        images = kwargs.pop("images", None)
        scores = kwargs.pop("scores", None)
        times = kwargs.pop("times", None)
        heads = kwargs.pop("heads", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # if 'input_ids' in _inputs:
        #     print(_inputs["input_ids"])
        # else:
        #     print('No inputs id here')
        if images is not None:
            _inputs['images'] = images
        if times is not None:
            _inputs['times'] = times
        if scores is not None:
            _inputs['scores'] = scores
        if heads is not None:
            # switch the lm head if encounter <sync>
            if "input_ids" in _inputs:
                new_tokens = _inputs["input_ids"][:, -1]
                for batch_idx, new_token in enumerate(new_tokens):
                    if int(new_token) in self.swap_tokens:
                        heads[batch_idx] = self.swap_tokens[int(new_token)]

            _inputs['heads'] = heads
        # print(_inputs['heads'])
        # print(_inputs['heads'])
        return _inputs


AutoConfig.register("trace_mistral", TraceMistralConfig)
AutoModelForCausalLM.register(TraceMistralConfig, TraceMistralForCausalLM)
