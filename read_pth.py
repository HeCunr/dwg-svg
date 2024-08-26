import torch

f = "/tmp/deepsvg-train/pretrained/my_pth/ordered_60.pth"#"/tmp/deepsvg-train/pretrained/my_pth/60.pth","/tmp/deepsvg-train/pretrained/hierarchical_ordered.pth.tar
#/tmp/deepsvg-train/pretrained/my_pth/ordered_60.pth
state_dict = torch.load(f)
for key in state_dict:
    print(key)
  #  print(state_dict[key].shape)
    print(state_dict[key])
    print()