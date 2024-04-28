import folder_paths
import json
import os
import re
from PIL import Image
from jsonpath import jsonpath


def webinfo(metadata):
    # 使用正则表达式查找对应的值
    text = "Positive prompt:" + str(metadata["parameters"])
    positive_pattern = r"Positive prompt:([\s\S]*)Negative prompt:"
    negative_pattern = r"Negative prompt:([\s\S]*)Steps:"
    steps_pattern = r"Steps: (\d+)"
    seed_pattern = r"Seed: (\d+)"
    cfg_pattern = r"CFG scale: (\d+)"

    # 搜索并提取对应的值
    positive_match = re.findall(positive_pattern, text)
    negative_match = re.findall(negative_pattern, text)
    steps_match = re.search(steps_pattern, text)
    seed_match = re.search(seed_pattern, text)
    cfg_match = re.search(cfg_pattern, text)
    # 存储提取的值
    positive = positive_match[0] if negative_match else None
    negative = negative_match[0] if negative_match else None
    steps = steps_match.group(1) if steps_match else None
    seed = seed_match.group(1) if seed_match else None
    cfg = cfg_match.group(1) if cfg_match else None
    allinfo=positive+negative+steps+seed+cfg
    return allinfo,positive, negative, steps, seed, cfg


def comfyinfo(metadata):
    data = json.loads(metadata["prompt"])
    steps = jsonpath(data, "$..steps")
    seed = jsonpath(data, "$..seed")
    cfg = jsonpath(data, "$..cfg")
    posid = jsonpath(data, "$..positive")
    negid = jsonpath(data, "$..negative")
    positive = jsonpath(data, "$.." + list(posid[0])[0] + "..text")  # list(posid[0])[0]控制选择多个参数
    negative = jsonpath(data, "$.." + list(negid[0])[0] + "..text")
    allinfo = "steps:" + str(steps) + "\n" \
              "seed:" + str(seed) + "\n" \
              "cfg:" + str(cfg) + "\n" \
              "positive" + str(positive) + "\n" \
              "negative:" + str(negative)
    return allinfo, positive, negative, steps, seed, cfg


# 0 定义一个类，一个节点就是一个类。comfyui会引入这个类作为一个节点==============================
class PngInfoBXDSJS:
    def __init__(self):
        pass

    # 1 定义这个节点在ui界面中的位置=======================================================
    CATEGORY = "Pretreatment"

    # ====================定义输入===========================================================
    @classmethod
    def INPUT_TYPES(s):  # 固定格式，输入参数种类
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # 返回一个包含所需输入类型的字典
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            },

        }

    # 4 右边的输出点在这里定义=============================================================
    OUTPUT_NODE = True  # 表明它是一个输出节点
    # 输出的数据类型，需要大写
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "FLOAT",)
    # 自定义输出名称
    RETURN_NAMES = ("all", "positive", "negative", "steps", "cfg",)

    # 5 节点的核心功能逻辑在这里定义========================================================
    FUNCTION = "pnginfo"  # 核心功能函数名称，将运行这个类中的这个方法

    def pnginfo(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        # 读取PNG图像
        img = Image.open(image_path)
        # 获取元数据
        metadata = img.info
        if "parameters" in metadata:
            allinfo,positive, negative, steps, seed, cfg = webinfo(metadata)
            return allinfo, positive, negative, steps, seed, cfg
        elif "prompt" in metadata:
            allinfo,positive, negative, steps, seed, cfg = comfyinfo(metadata)
            return allinfo, positive[0], negative[0], steps[0], seed[0], cfg[0]

    # 注释：实现了以上结构，就能在comfyui中引入这个节点，但这个节点现在是不能工作的
