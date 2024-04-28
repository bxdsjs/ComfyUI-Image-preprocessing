# import torchvision.transforms as T
# import torch
# from PIL import Image, ImageSequence, ImageOps
# import numpy as np


class CenterCrop:
    def __init__(self):
        self.compress_level = 4

    CATEGORY = "Pretreatment"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,  # 默认
                    "min": 1,
                    "max": 10000,
                    "step": 2,  # 步长
                    "display": "number"}),
                "height": ("INT", {
                    "default": 512,  # 默认
                    "min": 1,
                    "max": 10000,
                    "step": 2,  # 步长
                    "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    def crop(self, image,width,height):
        d1, d2, d3, d4 = image.size()
        output_image=image[:,int(d2/2-height):int(d2/2+height),int(d3/2-width):int(d3/2+width),:]
        return (output_image,)
