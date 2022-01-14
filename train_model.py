#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import sys
import os
import argparse
from torchvision import datasets, transforms
import logging
import warnings
from PIL import ImageFile

os.system("pip install smdebug")

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, device, test_loader):
    loss_optim = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_optim(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
              
    test_loss_avg = test_loss / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(test_loss_avg, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset))
    )
    
    return test_loss_avg

def train(args, model, train_loader, test_loader, optimizer):
    if args.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    hook = get_hook(create_if_not_exists=True)
    loss_optim = args.criterion
    if hook:
        hook.register_loss(loss_optim)
    
    for epoch in range(args.epochs):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            print("TRAINING...with {} epochs, batch_idx:{}, len(train_loader):{}".format(args.epochs, batch_idx, len(train_loader)))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_optim(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        if hook:
            hook.set_mode(modes.EVAL)
        print("TESTING...")
        test_loss = test(model, device, test_loader)
            
        logger.info("Epoch: {} train loss: {}, test loss: {}".format(epoch, train_loss, test_loss))
    save_model(model, args.model_dir)
    
def net():
    # create model
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
        
    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = net.fc.in_features
    
    net.fc = nn.Linear(fc_inputs, 2)
            
    #net.to(device)
    
    return net

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def create_data_loaders(data, batch_size):
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )

def main(args):
    if args.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = args.criterion
    print("args.lr: {}".format(args.lr))
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # download from s3 to data_dir
    logger.info("Fetching dataset from s3")
    
    os.system("aws s3 cp s3://sagemaker-us-east-1-807116804612/image_classification/dogs-vs-cats_smaller_dataset/ dogs-vs-cats_smaller_dataset --recursive")
    
    logger.info("Fetched dataset from s3")
                
    data_dir = 'dogs-vs-cats_smaller_dataset'
    
    trainset = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose(
        [   
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ))
    
    testset = datasets.ImageFolder(os.path.join(data_dir, "val"), transforms.Compose(
        [   
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ))
    
    # create train and test loaders
    train_loader = create_data_loaders(trainset, args.batch_size)
    
    test_loader = create_data_loaders(testset, args.batch_size)
    
    train(args, model, train_loader, test_loader, optimizer)
    
    # torch.save(model, 'model_save')

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("--gpu", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--criterion", type=str, default=nn.CrossEntropyLoss())
    
    args=parser.parse_args()
    
    main(args)
