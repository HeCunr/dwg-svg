import torch
from collections import OrderedDict

def convert_to_ordereddict(model_path, output_path):
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Convert the checkpoint to OrderedDict format
    ordered_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        ordered_checkpoint[key] = value

    # Create the final structure
    model = OrderedDict()
    model['model'] = ordered_checkpoint

    # Save the OrderedDict
    torch.save(model, output_path)

# Specify the input and output paths
input_model_path = "/tmp/deepsvg-train/pretrained/my_pth/60.pth"
output_model_path = "/tmp/deepsvg-train/pretrained/my_pth/ordered_60.pth"

# Convert and save the model
convert_to_ordereddict(input_model_path, output_model_path)
