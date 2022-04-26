import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Nets import ResBlock1, ResNet1_fine2, ResNet1_fine, ResNet1_coarse, restriction_matrix, prolongation_matrix, prolongation, restriction

dim_in = 28*28
dim_out = 10
reslayer_size = 100#100
no_reslayers_fine2 = 5
no_reslayers_fine = 3
no_reslayers = int((no_reslayers_fine+1)/2) # coarse

resnet_fine2 = ResNet1_fine2(dim_in, reslayer_size, dim_out, ResBlock1, h=0.25)
resnet_fine = ResNet1_fine(dim_in,reslayer_size,dim_out,ResBlock1,h=0.5)
resnet_coarse = ResNet1_coarse(dim_in,reslayer_size,dim_out,ResBlock1,h=1)

device = 'cpu'

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
batch_size = 70#30 #60

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


#for X, y in test_dataloader:
#    print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {y.shape} {y.dtype}")
#    break


model = resnet_fine
model_fine2 = resnet_fine2
model_fine = resnet_fine
model_coarse = resnet_coarse

loss_fn = nn.CrossEntropyLoss()
loss_fn_fine = loss_fn
loss_fn_coarse = nn.CrossEntropyLoss() # the modified loss fun for multilevel will be defined during training!
# #this loss function is needed for the gradients!


optimizer_fine = torch.optim.SGD(model_fine.parameters(), lr=1e-3)
optimizer_coarse = torch.optim.SGD(model_coarse.parameters(), lr=1e-3)

N1, N2, N3, N4 = 1,1,1,1

## 2-level nested iteration and mu-cycle
def train_2level(dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine, optimizer_coarse, lr,no_reslayers_coarse, prolong_matrix = False, Print=False):
    toc = time.perf_counter()
    no_reslayers_fine = int(2* no_reslayers_coarse -1)
    if prolong_matrix:
        prolong = prolongation_matrix(reslayer_size, no_reslayers,dim_in,dim_out, sparse = True)
        restr = torch.t(prolong)
    size = len(dataloader.dataset)
    model_fine.train()
    model_coarse.train()
    for batch, (X, y) in enumerate(dataloader):
        if batch % 100 ==0:
            toc2 = time.perf_counter()
        X, y = X.to(device), y.to(device)

        # Nested iteration (N1 iterations)
        ## iterate N1 times on coarse grid
        for i in range(N1):
            # Compute prediction error
            pred = model_coarse(X)
            loss_coarse = loss_fn_coarse(pred, y)

            # Backpropagation
            optimizer_coarse.zero_grad()
            loss_coarse.backward()
            optimizer_coarse.step()

        ## prolongate to fine grid
        vec_coarse = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        if prolong_matrix:
            vec_fine = torch.mv(prolong,vec_coarse)
        else:
            vec_fine = prolongation(vec_coarse, reslayer_size, no_reslayers_coarse, dim_in, dim_out)
        torch.nn.utils.vector_to_parameters(vec_fine,model_fine.parameters())

        #start V-cycle
        ##0) iterate on fine level N2 times
        for i in range(N2):
            # Compute prediction error
            pred = model_fine(X)
            loss_fine = loss_fn_fine(pred, y)

            # Backpropagation
            optimizer_fine.zero_grad()
            loss_fine.backward()
            optimizer_fine.step()
        x1 = torch.nn.utils.parameters_to_vector(model_fine.parameters())

        ###1) compute gradient on fine level #and restrict to coarse level
        g = []
        for param in model_fine.parameters():
            g.append(param.grad.view(-1))
        g = torch.cat(g)

        ###4) restrict gradient to coarse level
        if prolong_matrix:
            g2 = torch.mv(restr,g)
        else:
            g2 = restriction(g,reslayer_size,no_reslayers_fine,dim_in,dim_out)

        ##2) restrict parameters to coarser level
        vec_fine = torch.nn.utils.parameters_to_vector(model_fine.parameters())
        if prolong_matrix:
            vec_coarse = torch.mv(restr, vec_fine)
        else:
            vec_coarse = restriction(vec_fine,reslayer_size, no_reslayers_fine, dim_in,dim_out)
        x1_bar = vec_coarse
        torch.nn.utils.vector_to_parameters(vec_coarse, model_coarse.parameters())

        ###3) compute gradient of loss function on the coarse level
        g1 = []
        for param in model_coarse.parameters():
            g1.append(param.grad.view(-1))
        g1 = torch.cat(g1)

        ## 6) todo:  get new objective (instead of loss) function
        ### construct additional summand of new objective
        def loss_fn_coarse_mod(pred,y):
            '''paras_flat = []
            for param in model_coarse.parameters():
                print(param.size)
                paras_flat.append(param.flatten())
            paras_flat = torch.cat(paras_flat)
            print(paras_flat.shape)'''
            paras_flat = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            nu = torch.sub(g2, g1)
            # new loss = old loss + torch.dot(nu,paras_flat) # how do we need to define a loss function?
            loss = torch.sub(loss_fn_coarse(pred,y),torch.dot(nu,paras_flat))
            return loss

        ##7) iterate on coarse level N3 times (with new objectives)
        for i in range(N3):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_coarse_mod(pred, y)

            # Backpropagation
            optimizer_coarse.zero_grad()
            loss.backward()
            optimizer_coarse.step()

        ### 8) compute prolongation of difference
        x2_bar = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        e2_bar = torch.sub(x2_bar,x1_bar)
        if prolong_matrix:
            e2 = torch.mv(prolong,e2_bar)
        else:
            e2 = prolongation(e2_bar,reslayer_size,no_reslayers_coarse,dim_in,dim_out)

        ## 9) update fine weights (maybe with line search)
        ## for now, without line search
        x2 = torch.sub(x1,e2,alpha=-lr)
        torch.nn.utils.vector_to_parameters(x2,model_fine.parameters())

        ## 10) iterate on fine level N4 times
        for i in range(N4):
            # Compute prediction error
            pred = model_fine(X)
            loss_fine = loss_fn_fine(pred, y)

            # Backpropagation
            optimizer_fine.zero_grad()
            loss_fine.backward()
            optimizer_fine.step()

        if batch % 100 == 0:
            current = batch*len(X)
            #loss = loss_fn_fine.item()
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if Print:
                print('iteration no. ', batch*len(X))
                print('loss',loss_fine.item())
            tic2 = time.perf_counter()
            if Print:
                print('needed time for this batch: ', tic2 - toc2)
    tic = time.perf_counter()
    if Print:
        print('needed time for one epoch: ', tic-toc)



## classical training (minibatch sgd)
def train_classical(dataloader, model, loss_fn, optimizer, Print=False):
    toc = time.perf_counter()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        if batch % 100 ==0:
            toc2 = time.perf_counter()

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if Print:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            tic2 = time.perf_counter()
            if Print:
                print('needed time for this batch: ', tic2-toc2)
            #model.layer1.l1.weight = torch.nn.parameter.Parameter(data=torch.ones(10,10))
            #model.layer1.l1.weight = model.layer2.l1.weight
            #print('iteration no. ', batch*len(X))
    tic = time.perf_counter()
    if Print:
        print('needed time for one epoch: ', tic-toc)

# can stay
def test(dataloader, model, loss_fn, Print = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if Print:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

print('First classical training!')
toc = time.perf_counter()
epochs = 1 #2#5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_classical(train_dataloader, model, loss_fn, optimizer_fine,Print=True)
    #train_classical(train_dataloader, model_fine2, loss_fn_fine, optimizer_fine, print=True)
    correct = test(test_dataloader, model, loss_fn, Print=True)
    #test(test_dataloader, model_fine2, loss_fn_fine, print=True)
tic = time.perf_counter()
print('Needed time for the whole classical training: ', tic-toc)
#print('Now we look at state_dict.')
#for key in model.state_dict():
    #print(key)
    #print(model.state_dict()[key])
#print('weights  w1 in training loop of first residual block: ', model.state_dict().keys())
#grads = []
#paras = []
#print('model.parameters',torch.nn.utils.parameters_to_vector(model.parameters()).size())
#for param in model.parameters():
    #grads.append(param.grad.view(-1))
    #print(param.size())
    #print(param)
    #paras.append(param.flatten())
#grads = torch.cat(grads)
#paras = torch.cat(paras)
#print(grads.shape)
#print(paras.shape)
#print(grads)


## multilevel training:
print('Now 2-level-training!')
toc = time.perf_counter()
lr = 1e-3
no_reslayers_coarse=2
no_reslayers_fine = 3
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_2level(train_dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine, optimizer_coarse, lr, no_reslayers_coarse, Print=True)
    #train_2level(train_dataloader, model_fine2, model_fine, loss_fn_fine, loss_fn_coarse, optimizer_fine,
    #             optimizer_coarse, lr, no_reslayers_fine, print=True)
    correct = test(test_dataloader, model_fine, loss_fn_fine, Print=True)
    #test(test_dataloader, model_fine2, loss_fn_fine, print=True)
tic = time.perf_counter()
print('Needed time for the whole 2-level training: ', tic-toc)
print("Done!")




## Now we train systematically with different batchsizes (all other hyperparameters are fixed)
## we write the results in a .txt file


with open("mytest_bs.txt", "w") as file1:
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes:")
    file1.write("lr = "+str(lr)+" and reslayersize = "+str(reslayer_size))
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.")
bs_list = [10,20,30,40,50,60]
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
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes:")
    file1.write("bs = "+str(bs)+" and reslayersize = "+str(reslayer_size))
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.")
lr_list = [1e-4,1e-3,1e-2]
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

lr = 1e-3
with open("mytest_rw.txt", "w") as file1:
    file1.write("Training the Net on MNIST data set, with res net with "+str(no_reslayers_fine)+" reslayers (on fine level) for different batchsizes and variable width of the reslayers:")
    file1.write("bs = "+str(bs)+" and lr = "+str(lr))
    file1.write("classical training runs 5 epochs, 2-level training runs 2 epochs. We measure the accuracy and the needed time.")
rw_list = [10,50,100,200,500,700]
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
        file1.write(f"Accuracy: {(100*correct):>0.1f}% \n")