import torch


class Pad:
    CATEGORY = "Pretreatment"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "expand_image"

    def expand_image(self, image, left, top, right, bottom):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4), dtype=torch.float32, ) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image
        return (new_image,)
