import time
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Nets import ResBlock1, ResNet1_fine2, ResNet1_fine #, ResNet1_coarse, prolongation, restriction, prolongation_matrix

from Nets2 import train_multilevel, gen_hierarch_models, train_classical, test

### Now we train the NN in different ways:
### We specify the data etc....

# we set a fixed seed for the batch generation
random.seed(1)

# set no of kernels to use
torch.set_num_threads(8) # uses 8 kernels (i have 8)

# we load the datasets:
# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#define batchsize for minibatch-SGD
batch_size = 70

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# we build the models:
dims = [28*28,10,10]
dim_in = dims[0] #28*28 # for images:X.shape[2]*X.shape[3]
dim_out = dims[2] #10
reslayer_size = dims[1] #10 #100

#for classical training
resnet_classical_3 = ResNet1_fine(dim_in,reslayer_size,dim_out,ResBlock1,h=0.5) # 3 reslayers
resnet_classical_5 = ResNet1_fine2(dim_in, reslayer_size, dim_out, ResBlock1, h=0.25) # 5 reslayers

loss_fn = nn.CrossEntropyLoss()

lr_classical = 1e-3
optimizer_3 = torch.optim.SGD(resnet_classical_3.parameters(), lr_classical)
optimizer_5 = torch.optim.SGD(resnet_classical_5.parameters(), lr_classical)

'''
print('First classical training!')

toc = time.perf_counter()
epochs = 1 #2#5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #train_classical(train_dataloader, resnet_classical_5, loss_fn, optimizer_5,Print=True)
    train_classical(train_dataloader, resnet_classical_3, loss_fn, optimizer_3, Print=True)
    #correct = test(test_dataloader, resnet_classical_5, loss_fn, Print=True)
    correct = test(test_dataloader, resnet_classical_3, loss_fn, Print=True)
tic = time.perf_counter()
print('Needed time for the whole classical training: ', tic-toc)
print(" \n") '''


## multilevel training:

'''#for multi-level training
resnet_fine2 = ResNet1_fine2(dim_in, reslayer_size, dim_out, ResBlock1, h=0.25) # 5 reslayers
resnet_fine = ResNet1_fine(dim_in,reslayer_size,dim_out,ResBlock1,h=0.5) # 3 reslayers
resnet_coarse = ResNet1_coarse(dim_in,reslayer_size,dim_out,ResBlock1,h=1) # 2 reslayers

loss_fn_fine2 = nn.CrossEntropyLoss()
loss_fn_fine = nn.CrossEntropyLoss()
loss_fn_coarse = nn.CrossEntropyLoss() # the modified loss fun for multilevel will be defined during training!
# #this loss function is needed for the gradients and used in building the new loss!

lr_coarse = 1e-3
lr_fine = 1e-3
lr_fine2 = 1e-3
optimizer_fine2 = torch.optim.SGD(resnet_fine2.parameters(), lr_fine2)
optimizer_fine = torch.optim.SGD(resnet_fine.parameters(), lr_fine)
optimizer_coarse = torch.optim.SGD(resnet_coarse.parameters(), lr_coarse)

no_reslayers_coarse = 2
no_reslayers_fine = 3

iteration_numbers = [1,1,1,1]
lr_xtra = 1e-3 '''

print('Now multilevel-training!')
toc = time.perf_counter()
no_levels = 2
coarse_no_reslayers = 2
iteration_numbers = [1,1,1,1]
lr = 1e-3
model_list = gen_hierarch_models(no_levels, coarse_no_reslayers, dims, ResBlock1)
loss_fns = []
for i in range(no_levels):
    loss_fns.append(nn.CrossEntropyLoss())
optimizers = []
for i in range(no_levels):
    optimizers.append(torch.optim.SGD(model_list[i].parameters(), lr))
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_multilevel(train_dataloader,model_list, loss_fns, optimizers, lr, iteration_numbers,no_levels,coarse_no_reslayers,dims, Print=True) #3-2 layers
    #train_multilevel(train_dataloader,model_list, loss_fns, optimizers, lr, iteration_numbers,no_levels,coarse_no_reslayers,dims, Print=True) # 5-3 layers
    correct = test(test_dataloader, model_list[no_levels-1], loss_fns[no_levels-1], Print=True)
    #correct = test(test_dataloader, model_list[no_levels-1], loss_fns[no_levels-1], Print=True)
tic = time.perf_counter()
print('Needed time for the whole 2-level training: ', tic-toc)
print("Done!")

