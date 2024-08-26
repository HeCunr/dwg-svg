import os
os.chdir("..")
#%%


import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
from IPython.display import display
from deepsvg.svglib.svg import SVG

from deepsvg import utils
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.utils import to_gif
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from deepsvg.utils.utils import batchify, linear




#%%
pretrained_path = "/tmp/deepsvg-train/pretrained/my_pth/ordered_60.pth"  #"./pretrained/hierarchical_ordered.pth.tar"#
from configs.deepsvg.hierarchical_ordered import Config
print(0)

cfg = Config()
cfg.model_cfg.dropout = 0.  # for faster convergence
model = cfg.make_model().to(device)
utils.load_model(pretrained_path, model)
model.eval()
dataset = load_dataset(cfg)
print(1)

def encode(data):
    model_args = batchify((data[key] for key in cfg.model_args), device)
    with torch.no_grad():
        z = model(*model_args, encode_mode=True)
        return z

def encode_svg(svg):
    data = dataset.get(svg=svg)
    return encode(data)
def load_svg(filename):
    svg = SVG.load_svg(filename)
    svg = dataset.simplify(svg)
    svg = dataset.preprocess(svg, mean=True)
    return svg

def decode(z, do_display=True, return_svg=False, return_png=False):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256), allow_empty=True).normalize().split_paths().set_color("random")

    if return_svg:
        return svg_path_sample

    return svg_path_sample.draw(do_display=do_display, return_png=return_png)

print(2)
lego = load_svg("/tmp/deepsvg-train/dataset/svgs_simplified/0.svg")

print(3)
z = encode_svg(lego)

print(4)
print(z)

print(5)


