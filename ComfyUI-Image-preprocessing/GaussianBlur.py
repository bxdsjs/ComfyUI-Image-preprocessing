import torchvision.transforms as T
import torch
from PIL import Image, ImageSequence, ImageOps
import numpy as np


class GaussianBlur:
    def __init__(self):
        self.compress_level = 4

    CATEGORY = "Pretreatment"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {
                    "default": 0,  # 默认
                    "min": 0,
                    "max": 16384,
                    "step": 1,  # 步长
                    "display": "number"}),
                "sigma": ("INT", {
                    "default": 0,  # 默认
                    "min": 0,
                    "max": 16384,
                    "step": 2,  # 步长
                    "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gaussian"

    def gaussian(self, image, kernel_size, sigma):

        global img
        output_images = []
        for (batch_number, image) in enumerate(image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        img = T.GaussianBlur(kernel_size, sigma)(img)

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)

        else:
            output_image = output_images[0]
        return (output_image,)
