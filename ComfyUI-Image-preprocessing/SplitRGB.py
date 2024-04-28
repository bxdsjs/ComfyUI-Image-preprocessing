import torchvision.transforms as T
import torch
from PIL import Image, ImageSequence, ImageOps
import numpy as np


class SplitRGB:
    def __init__(self):
        self.compress_level = 4

    CATEGORY = "Pretreatment"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "R": ("BOOLEAN", {"default": True}),
                "G": ("BOOLEAN", {"default": True}),
                "B": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "splitRGB"

    def splitRGB(self, image, R, G, B):

        global img
        output_images = []
        for (batch_number, image) in enumerate(image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        r, g, b = img.split()

        if not R:
            r = r.point(lambda i: i == 0)
        if not G:
            g = g.point(lambda i: i == 0)
        if not B:
            b = b.point(lambda i: i == 0)
        img = Image.merge("RGB", (r, g, b))

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
