import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

from get_input_args import get_input_args

def process_image(image):
    ''' Scales, crops, and normalizes an image for a model,
        returns a Numpy array
    '''
    with Image.open(image) as im:
        im.thumbnail((256, 256))
        crop_width = 224
        crop_height = 224
        img_width, img_height = im.size
        im = im.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
        im = np.float32(im) / np.max(im)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std
        im = im.transpose(2, 0, 1)
        
        return im

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained model
    '''
    model.eval()

    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.to(torch.float)
    img = img.to(device)
    img = img.view(1, 3, 224, 224)

    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)
    
    return torch.topk(ps, topk)

def main():
    in_arg = get_input_args('predict')
    print(in_arg)

    # Rebuild model
    print("Rebuilding model...")
    checkpoint = torch.load(in_arg.checkpoint)

    model = eval(f"models.{checkpoint['arch']}(pretrained=True)")
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=checkpoint['dropout'])),
        ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Enable CUDA acceleration if enabled and available
    if in_arg.gpu:
        if torch.cuda.is_available():
            print("CUDA acceleration enabled")
            device = torch.device("cuda:0")
        else:
            print("GPU not available!")
            device = torch.device("cpu")
    else:
        print("CUDA acceleration not enabled")
        device = torch.device("cpu")

    model.to(device);

    # Prediction
    print("Starting prediction...")
    probs, classes = predict(in_arg.input_image, model, device)
    probs, classes = probs.to('cpu').numpy()[0], classes.to('cpu').numpy()[0]
    
    # Class translation
    class_mapper = in_arg.category_names
    with open(class_mapper, 'r') as f:
        cat_to_name = json.load(f)
    image_classes = [cat_to_name[str(cls+1)].title() for cls in classes]

    # Top Prediction
    print(f"\n***Top Prediction***\n"
          f"{image_classes[0]} - {probs[0]}\n")

    # Top K
    if in_arg.top_k <= len(image_classes):
        top_k = in_arg.top_k
    else:
        top_k = len(image_classes)

    print(f"***The Top {top_k} Image Classes***")
    for i in range(top_k):
        print(f"{image_classes[i]} - {probs[i]}")

if __name__ == "__main__":
    main()