import torch
from PIL import Image, ImageSequence, ImageOps
import numpy as np
import math


class ReplaceRGB:
    def __init__(self):
        self.compress_level = 4

    CATEGORY = "Pretreatment"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_source": ("STRING", {
                    "default": "#000000",  # 默认存在内容
                    "multiline": False}),  # 是否允许多行输入
                "tolerance": ("INT", {"default": 0, "min": 0, "max": 800, "step":1}),
                "color_target": ("STRING", {
                    "default": "#000000",  # 默认存在内容
                    "multiline": False}),  # 是否允许多行输入
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replaceRGB"

    # ======将图片分成小方块=======
    def Hex_to_RGB(self, inhex):
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgbval = (int(rval, 16), int(gval, 16), int(bval, 16))
        return rgbval

    def ColourDistance(self, rgb_1, rgb_2):
        R_1, G_1, B_1 = rgb_1
        R_2, G_2, B_2 = rgb_2
        rmean = (R_1 + R_2) / 2
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2
        return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

    def replaceRGB(self, image, color_source, tolerance, color_target):

        global img
        output_images = []
        for (batch_number, image) in enumerate(image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))  # tensor=>pil

        img_array = img.load()
        # 遍历每一个像素块，并处理颜色
        width, height = img.size  # 获取宽度和高度
        for x in range(0, width):
            for y in range(0, height):
                rgb = img_array[x, y]  # 获取一个像素块的rgb
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                if self.ColourDistance(img_array[x, y], self.Hex_to_RGB(color_source)) <= tolerance:  # 判断规则
                    img_array[x, y] = self.Hex_to_RGB(color_target)

        for i in ImageSequence.Iterator(img):  # pil=>tensor
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
