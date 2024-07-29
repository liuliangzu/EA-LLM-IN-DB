#importing library
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, get_peft_model, PeftModel
from dataset_prepare import *
import torch, os
from transformers import PhiForCausalLM
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from typing import List, Optional, Tuple, Union


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

def constrained_decoding(logits, allowed_token_ids):
    mask = torch.ones(logits.size(-1), dtype=torch.bool)
    mask[allowed_token_ids] = False
    logits.masked_fill_(mask, -float('inf'))
    return logits

def constrained_decoding_full(logits, mask, output_lens, not_allowed_token_ids):
    for idx in range(len(output_lens)):
        # mask = torch.ones_like(logits[idx, -1-output_lens[idx]:-1,]).bool()
        # mask[:,allowed_token_ids] = False
        logits[idx, -1-output_lens[idx]:-1,not_allowed_token_ids] = -float('inf')
    return logits


class new_PhiForCausalLM(PhiForCausalLM):
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
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_len=None,
            mask=None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            r"""
            Args:
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            Returns:

            Example:

            ```python
            >>> from transformers import AutoTokenizer, PhiForCausalLM

            >>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1")
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

            >>> prompt = "This is an example script ."
            >>> inputs = tokenizer(prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            'This is an example script .\n\n\n\nfrom typing import List\n\ndef find_most_common_letter(words: List[str'
            ```"""

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            # for idx in range(input_ids.shape[1]):
            #     outputs = self.model(
            #         input_ids=input_ids[:,idx].unsqueeze(1),
            #         attention_mask=attention_mask,
            #         position_ids=position_ids,
            #         past_key_values=past_key_values,
            #         inputs_embeds=inputs_embeds,
            #         use_cache=use_cache,
            #         output_attentions=output_attentions,
            #         output_hidden_states=output_hidden_states,
            #         return_dict=return_dict,
            #         cache_position=cache_position,
            #     )
                
            #     output_hidden_states = outputs.last_hidden_state
            #     print()
            
            
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
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if output_len is not None:
                    logits = constrained_decoding_full(logits, mask, output_len, self.not_allowed_token_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                logits[:, -1,self.not_allowed_token_ids] = -float('inf')

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )