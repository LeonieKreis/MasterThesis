import time
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Nets import ResBlock1, ResNet1_fine2, ResNet1_fine, ResNet1_coarse, prolongation, restriction, prolongation_matrix
from Nets2 import train_2level, train_classical, test


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
dim_in = 28*28 # for images:X.shape[2]*X.shape[3]
dim_out = 10
reslayer_size = 10 #100
#no_reslayers_fine2 = 5
#no_reslayers_fine = 3
#no_reslayers = int((no_reslayers_fine+1)/2) # coarse

#for classical training
resnet_classical_3 = ResNet1_fine(dim_in,reslayer_size,dim_out,ResBlock1,h=0.5) # 3 reslayers
resnet_classical_5 = ResNet1_fine2(dim_in, reslayer_size, dim_out, ResBlock1, h=0.25) # 5 reslayers

loss_fn = nn.CrossEntropyLoss()

lr_classical = 1e-3
optimizer_3 = torch.optim.SGD(resnet_classical_3.parameters(), lr_classical)
optimizer_5 = torch.optim.SGD(resnet_classical_5.parameters(), lr_classical)


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
print(" \n")


## multilevel training:

#for 2-level training
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
lr_xtra = 1e-3

print('Now 2-level-training!')
toc = time.perf_counter()

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_2level(train_dataloader, resnet_fine, resnet_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine, optimizer_coarse, lr_xtra, iteration_numbers, no_reslayers_coarse, Print=True) #3-2 layers
    #train_2level(train_dataloader, resnet_fine2, resnet_fine, loss_fn_fine2, loss_fn_fine, optimizer_fine2,
                 #optimizer_fine, lr_xtra, no_reslayers_fine, Print=True) # 5-3 layers
    correct = test(test_dataloader, resnet_fine, loss_fn_fine, Print=True)
    #correct = test(test_dataloader, resnet_fine2, loss_fn_fine2, Print=True)
tic = time.perf_counter()
print('Needed time for the whole 2-level training: ', tic-toc)
print("Done!")




## Now we train systematically with different batchsizes (all other hyperparameters are fixed)
## we write the results in a .txt file

train_systematically = False
if train_systematically == True:
    #train systematically #->todo
    x = 0


'''with open("mytest_bs.txt", "w") as file1:
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes:\n")
    file1.write("lr = "+str(lr)+" and reslayersize = "+str(reslayer_size)+"\n")
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.\n")
bs_list = [10,20,30,40,50,60,70,80,90,100]
for bs in bs_list:
    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)
    #classical training
    toc = time.perf_counter()
    epochs = 5  # 2#5
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_classical(train_dataloader, model, loss_fn, optimizer_fine)
        # train_classical(train_dataloader, model_fine2, loss_fn_fine, optimizer_fine)
        correct = test(test_dataloader, model, loss_fn)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text = 'Needed time for the whole classical training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_bs.txt", "a") as file1:
        # Writing data to a file
        file1.write("Batch size: "+str(bs)+"\n")
        file1.write("Classical training: ")
        file1.write(text)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")

    # 2-level training:
    toc = time.perf_counter()
    lr = 1e-3
    no_reslayers_coarse = 2
    # no_reslayers_fine = 3
    epochs = 2
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_2level(train_dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine,
                     optimizer_coarse, lr, no_reslayers_coarse)
        # train_2level(train_dataloader, model_fine2, model_fine, loss_fn_fine, loss_fn_coarse, optimizer_fine,
        #             optimizer_coarse, lr, no_reslayers_fine)
        correct = test(test_dataloader, model_fine, loss_fn_fine)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text2 = 'Needed time for the whole 2-level training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_bs.txt", "a") as file1:
        # Writing data to a file
        file1.write("Batch size: "+str(bs) + "\n")
        file1.write("2-level training: ")
        file1.write(text2)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")




## Now we train systematically with different leraning rates (all other hyperparameters are fixed)
## we write the results in a .txt file

bs = 40
with open("mytest_lr.txt", "w") as file1:
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes:\n")
    file1.write("bs = "+str(bs)+" and reslayersize = "+str(reslayer_size)+"\n")
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.\n")
lr_list = [1e-4,1e-3,5e-3,1e-2,2e-2,5e-2,1e-1]
for lr in lr_list:
    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)
    optimizer_fine = torch.optim.SGD(model_fine.parameters(), lr=lr)
    optimizer_coarse = torch.optim.SGD(model_coarse.parameters(), lr=lr)
    #classical training
    toc = time.perf_counter()
    epochs = 5  # 2#5
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_classical(train_dataloader, model, loss_fn, optimizer_fine)
        # train_classical(train_dataloader, model_fine2, loss_fn_fine, optimizer_fine)
        correct = test(test_dataloader, model, loss_fn)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text = 'Needed time for the whole classical training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_lr.txt", "a") as file1:
        # Writing data to a file
        file1.write("learning rate: "+str(lr)+"\n")
        file1.write("Classical training: ")
        file1.write(text)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")

    # 2-level training:
    toc = time.perf_counter()
    no_reslayers_coarse = 2
    # no_reslayers_fine = 3
    epochs = 2
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_2level(train_dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine,
                     optimizer_coarse, lr, no_reslayers_coarse)
        # train_2level(train_dataloader, model_fine2, model_fine, loss_fn_fine, loss_fn_coarse, optimizer_fine,
        #             optimizer_coarse, lr, no_reslayers_fine)
        correct = test(test_dataloader, model_fine, loss_fn_fine)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text2 = 'Needed time for the whole 2-level training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_lr.txt", "a") as file1:
        # Writing data to a file
        file1.write("Learning rate: "+str(lr) + "\n")
        file1.write("2-level training: ")
        file1.write(text2)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")



## Now we train systematically with different reslayer widths (all other hyperparameters are fixed)
## we write the results in a .txt file

lr = 1e-2
with open("mytest_rw.txt", "w") as file1:
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes and variable width of the reslayers:\n")
    file1.write("bs = "+str(bs)+" and lr = "+str(lr)+"\n")
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.\n")
rw_list = [10,50,100,200,500,700,1000,2000]
for rw in rw_list:
    resnet_fine = ResNet1_fine(dim_in, rw, dim_out, ResBlock1, h=0.5)
    resnet_coarse = ResNet1_coarse(dim_in, rw, dim_out, ResBlock1, h=1)

    train_dataloader = DataLoader(training_data, batch_size=bs)
    test_dataloader = DataLoader(test_data, batch_size=bs)
    #classical training
    toc = time.perf_counter()
    epochs = 5  # 2#5
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_classical(train_dataloader, model, loss_fn, optimizer_fine)
        # train_classical(train_dataloader, model_fine2, loss_fn_fine, optimizer_fine)
        correct = test(test_dataloader, model, loss_fn)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text = 'Needed time for the whole classical training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_rw.txt", "a") as file1:
        # Writing data to a file
        file1.write("width of reslayers: "+str(rw)+"\n")
        file1.write("Classical training: ")
        file1.write(text)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")

    # 2-level training:
    toc = time.perf_counter()
    no_reslayers_coarse = 2
    # no_reslayers_fine = 3
    epochs = 2
    for t in range(epochs):
        #print(f"Epoch {t + 1}\n-------------------------------")
        train_2level(train_dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine,
                     optimizer_coarse, lr, no_reslayers_coarse)
        # train_2level(train_dataloader, model_fine2, model_fine, loss_fn_fine, loss_fn_coarse, optimizer_fine,
        #             optimizer_coarse, lr, no_reslayers_fine)
        correct = test(test_dataloader, model_fine, loss_fn_fine)
        # test(test_dataloader, model_fine2, loss_fn_fine)
    tic = time.perf_counter()
    text2 = 'Needed time for the whole 2-level training: '+str(tic - toc)+"\n"
    # Writing to file
    with open("mytest_rw.txt", "a") as file1:
        # Writing data to a file
        file1.write("width of reslayers: "+str(rw) + "\n")
        file1.write("2-level training: ")
        file1.write(text2)
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")  '''