import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from pathlib import Path

from get_input_args import get_input_args

def main():
    in_arg = get_input_args('train')

    # Load data
    print("Loading data...")
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # Build model
    print("Building model...")
    model = eval(f"models.{in_arg.arch}(pretrained=True)")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, in_arg.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(in_arg.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    model.classifier = classifier

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
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    model.to(device);

    # Train model
    print("Training model...")
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                # Evaluation mode
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                # Training mode
                model.train()

    # Save model as checkpoint
    checkpoint_path = Path(in_arg.save_dir)
    checkpoint_name = f'checkpoint_{in_arg.arch}_{int(time.time())}.pth'
    print(f"Saving checkpoint as '{checkpoint_name}'...")

    checkpoint = {'arch': in_arg.arch,
                  'input_size': classifier.fc1.in_features,
                  'hidden_units': in_arg.hidden_units,
                  'output_size': classifier.fc2.out_features,
                  'dropout': classifier.dropout.p,
                  'epochs': epochs,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    torch.save(checkpoint, checkpoint_path / checkpoint_name)
    print("Done.")

if __name__ == "__main__":
    main()