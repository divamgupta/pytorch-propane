import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms

from pytorch_propane.models import Model 
from pytorch_propane.registry import registry

def get_CIFAR10(root="/tmp/"):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset

@registry.register_dataset("cifar_data_train")
def train_data(pupu):
    print("pupu is the best " , pupu )
    return get_CIFAR10()[2]


@registry.register_dataset("cifar_data_test")
def test_data():
    return get_CIFAR10()[3]



class CifarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x

@registry.register_model("cifar_model_1.0")
def get_model( some_arg ):
    print("some_arg" ,  some_arg )
    network = CifarNet()
    model = Model( network=network )
    model.compile( optimizer='adam' , loss='nll_loss' , cuda=True )
    return model 



from pytorch_propane.trainer import Trainer
tr = Trainer()
tr(model_name="cifar_model_1.0" , dataset_name="cifar_data_train" ,
  eval_dataset_name="cifar_data_test" , batch_size=12 , eval_batch_size=2 , pupu=45 
  , some_arg='jjj' )