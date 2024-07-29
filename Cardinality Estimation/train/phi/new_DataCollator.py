from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TextStreamer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import torch


class new_DataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], Mapping):
        #     batch = pad_without_fast_tokenizer_warning(
        #         self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        #     )
        # else:
        batch = {
            "input_ids": _torch_collate_batch([examples[idx]["input_ids"] for idx in range(len(examples))], self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
            "output_len":[example["output_len"] for example in examples],
            # "mask": torch.stack([examples[idx]["mask"] for idx in range(len(examples))])
        }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch