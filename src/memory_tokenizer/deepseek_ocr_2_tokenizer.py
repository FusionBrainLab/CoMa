from typing import List, TypedDict, Dict, Any
import base64
from pathlib import Path
import io
from dataclasses import dataclass
import math

import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image, ImageOps
from torchvision import transforms

from .memory_tokenizer import MemoryTokenizer
from ..agent_message import AgentMessage
from ..core.attr_access import AttrSetter

class DeepSeekOCR2Tokenizer(MemoryTokenizer):
    def __init__(self, *, model_name: str,
                        crop_mode: bool,
                        base_size: int,
                        image_size: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.crop_mode = crop_mode
        self.base_size = base_size
        self.image_size = image_size

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
            return best_ratio
        self.find_closest_aspect_ratio = find_closest_aspect_ratio

        def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            # print(target_ratios)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # print(target_aspect_ratio)
            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images, target_aspect_ratio
        self.dynamic_preprocess = dynamic_preprocess

    def __call__(self, *, memories: List[List[AgentMessage]]) -> Dict[str, Any]:
        message = memories[0][0]
        image_path = None
        prompt = None
        for content in message.content:
            if content.modality == 'image':
                image_path = content.content
            elif content.modality == 'text':
                prompt = content.content
        
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        images = [image]

        patch_size = 16
        downsample_ratio = 4

        valid_img_tokens = 0
        ratio = 1

        image_draw = image.copy()

        w,h = image_draw.size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    
        images_seq_mask = []

        image_token = '<image>'
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = self.tokenizer.encode(text_sep, add_special_tokens=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if self.crop_mode:
                if image.size[0] <= 768 and image.size[1] <= 768:
                    crop_ratio = [1, 1]
                else:
                    images_crop_raw, crop_ratio = self.dynamic_preprocess(image)
                
                global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                        color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5)))
                
                if self.base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif self.base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                
                images_list.append(self.image_transform(global_view))

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                if width_crop_num > 1 or height_crop_num > 1:
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(self.image_transform(images_crop_raw[i]))
                
                if self.image_size == 768:
                    valid_img_tokens += len(images_crop_list) * 144

                num_queries = math.ceil((self.image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((self.base_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (
                                num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if self.image_size <= 768:
                    print('directly resize')
                    image = image.resize((self.image_size, self.image_size))

                global_view = ImageOps.pad(image, (self.image_size, self.image_size),
                                        color=tuple(int(x * 255) for x in (0.5, 0.5, 0.5)))
                images_list.append(self.image_transform(global_view))

                if self.base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif self.base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif self.base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif self.base_size == 512:
                    valid_img_tokens += int(64 * 1)
                elif self.base_size == 768:
                    valid_img_tokens += int(144 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])

                num_queries = math.ceil((self.image_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries) * num_queries
                tokenized_image += [image_token_id]
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)

        """process the last text split"""
        tokenized_sep = self.tokenizer.encode(text_splits[-1], add_special_tokens=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str 
        images_seq_mask = [False] + images_seq_mask

        input_ids = torch.LongTensor(tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        
        output = {
            "input_ids": input_ids.unsqueeze(0).cuda(),
            "images":[(images_crop.cuda(), images_ori.cuda())],
            "images_seq_mask": images_seq_mask.unsqueeze(0).cuda(),
            "images_spatial_crop": images_spatial_crop
        }
        return output