from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import base64
import io
import json
from typing import Dict, List

import PIL.Image
import requests

import torch

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from ..smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag


def load_pil_images_pretrain(images: List[str]) -> List[PIL.Image.Image]:
    """Support file path or base64 images for pretraining data format.

    Args:
        text (str): the input text
        images (List[str]): list of image paths or base64 encoded image strings

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.
    """
    pil_images = []

    for image_data in images:
        if image_data.startswith("data:image"):
            # Image data is in base64 format
            _, image_data = image_data.split(",", 1)
            image_bytes = base64.b64decode(image_data)
            pil_img = PIL.Image.open(io.BytesIO(image_bytes))
        elif image_data.startswith("https://") or image_data.startswith("http://"):
            response = requests.get(image_data, timeout=10)
            response.raise_for_status()
            image_bytes = io.BytesIO(response.content)
            pil_img = PIL.Image.open(image_bytes)
        else:
            # Image data is a file path
            pil_img = PIL.Image.open(image_data)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)

    return pil_images


class XHSVLMLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose

        from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor

        self._load_model(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
    
    def _load_model(self, path: str):
        from transformers import AutoModel, AutoModelForCausalLM
        import sys
        sys.path.insert(0, path)

        import modeling_agi

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs = self._set_model_kwargs_torch_dtype(model_kwargs)
        logging.debug(f'using model_kwargs: {model_kwargs}')

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        import torch
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None,
            }.get(model_kwargs['torch_dtype'])
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype
        return model_kwargs


    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content, images = [], []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'text', 'text': "<image> "}
                images.append(s['value'])
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        
        prompt = "".join([i['text'] for i in content if i['type'] == 'text'])

        return prompt, load_pil_images_pretrain(images)

    def generate_inner(self, message, dataset=None):
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        content, images = self._prepare_content(message, dataset=dataset)
        messages.append({'role': 'user', 'content': content})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True)
        
        model_inputs = self.processor(prompt=text, images=images)
        input_ids = model_inputs.input_ids.unsqueeze(0).cuda()
        pixel_values = model_inputs.pixel_values.cuda()

        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            use_cache=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids_):] for input_ids_, output_ids in zip(input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response