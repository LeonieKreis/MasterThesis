import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from Nets import ResBlock1, ResNet1_fine, ResNet1_coarse, restriction_matrix, prolongation_matrix

dim_in = 28*28
dim_out = 10
reslayer_size = 10
no_reslayers_fine = 3
no_reslayers = (no_reslayers_fine+1)/2

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
#print(training_data)
batch_size = 60

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = resnet_fine
model_fine = resnet_fine
model_coarse = resnet_coarse

loss_fn = nn.CrossEntropyLoss()
loss_fn_fine = loss_fn
loss_fn_coarse = nn.CrossEntropyLoss() # the modified loss fun for multilevel will be defined during training!
# #this loss function is needed for the gradients!

#print(model.parameters)
optimizer_fine = torch.optim.SGD(model_fine.parameters(), lr=1e-3)
optimizer_coarse = torch.optim.SGD(model_coarse.parameters(), lr=1e-3)

N1, N2, N3, N4 = 1,1,2,1

## 2-level nested iteration and mu-cycle
def train_2level(dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine, optimizer_coarse, lr):
    size = len(dataloader.dataset)
    model_fine.train()
    model_coarse.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Nested iteration (N1 iterations)
        ## iterate N1 times on coarse grid
        for i in range(N1):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_coarse(pred, y)

            # Backpropagation
            optimizer_coarse.zero_grad()
            loss.backward()
            optimizer_coarse.step()
        ## prolongate to fine grid
        vec_coarse = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        vec_fine = torch.mm(prolongation_matrix(reslayer_size, no_reslayers,dim_in, dim_out, sparse=False),vec_coarse)
        torch.nn.utils.vector_to_parameters(vec_fine,model_fine.parameters())
        ''' model_fine.layer0.weight = model_coarse.layer0.weight
        model_fine.layer1.l1.weight = model_coarse.layer1.l1.weight
        model_fine.layer1.l1.bias = model_coarse.layer1.l1.bias
        model_fine.layer1.l2.weight = model_coarse.layer1.l2.weight
        s = model_fine.layer2.l1.size()[0]
        model_fine.layer2.l1.weight = torch.nn.parameter.Parameter(data=torch.zeros(s, s))
        model_fine.layer2.l1.bias = torch.nn.parameter.Parameter(data=torch.zeros(s))
        model_fine.layer.l2.weight = torch.nn.parameter.Parameter(data=torch.zeros(s, s))
        model_fine.layer3.l1.weight = model_coarse.layer2.l1.weight
        model_fine.layer3.l1.bias = model_coarse.layer2.l1.bias
        model_fine.layer3.l2.weight = model_coarse.layer2.l2.weight
        model_fine.layer4.weight = model_coarse.layer3.weight '''

        #start V-cycle
        ##0) iterate on fine level N2 times
        for i in range(N2):
            # Compute prediction error
            pred = model_fine(X)
            loss = loss_fn_fine(pred, y)

            # Backpropagation
            optimizer_fine.zero_grad()
            loss.backward()
            optimizer_fine.step()
        x1 = torch.nn.utils.parameters_to_vector(model_fine.parameters())
        ###1) compute gradient on fine level #and restrict to coarse level
        g = []
        for param in model_fine.parameters():
            g.append(param.grad.view(-1))
        g = torch.cat(g)
        ###4) restrict gradient to coarse level
        g2 = torch.mm(restriction_matrix(reslayer_size,no_reslayers, dim_in, dim_out,sparse=False),g)

        ##2) restrict parameters to coarser level
        vec_fine = torch.nn.utils.parameters_to_vector(model_fine.parameters())
        vec_coarse = torch.mm(restriction_matrix(reslayer_size,no_reslayers,dim_in, dim_out,sparse=False))
        x1_bar = vec_coarse
        torch.nn.utils.vector_to_parameters(vec_coarse, model_coarse.parameters())
        '''model_coarse.layer0.weight = model_fine.layer0.weight
        model_coarse.layer1.l1.weight = model_fine.layer1.l1.weight
        model_coarse.layer1.l1.bias = model_fine.layer1.l1.bias
        model_coarse.layer1.l2.weight = model_fine.layer1.l2.weight
        model_coarse.layer2.l1.weight = model_fine.layer3.l1.weight
        model_coarse.layer2.l1.bias = model_fine.layer3.l1.bias
        model_coarse.layer2.l2.weight = model_fine.layer3.l2.weight
        model_coarse.layer3.weight = model_fine.layer4.weight'''

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

        ### todo 8) compute prolongation of difference
        x2_bar = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        e2_bar = torch.sub(x2_bar,x1_bar)
        e2 = torch.mm(prolongation_matrix(reslayer_size,no_reslayers, dim_in, dim_out,sparse=False))

        ''''## prolongate the parameters to the finer level (not needed)
        model_fine.layer0.weight = model_coarse.layer0.weight
        model_fine.layer1.l1.weight = model_coarse.layer1.l1.weight
        model_fine.layer1.l1.bias = model_coarse.layer1.l1.bias
        model_fine.layer1.l2.weight = model_coarse.layer1.l2.weight
        s = model_fine.layer2.l1.size()[0]
        model_fine.layer2.l1.weight = torch.nn.parameter.Parameter(data=torch.zeros(s,s))
        model_fine.layer2.l1.bias = torch.nn.parameter.Parameter(data=torch.zeros(s))
        model_fine.layer.l2.weight = torch.nn.parameter.Parameter(data=torch.zeros(s,s))
        model_fine.layer3.l1.weight = model_coarse.layer2.l1.weight
        model_fine.layer3.l1.bias = model_coarse.layer2.l1.bias
        model_fine.layer3.l2.weight = model_coarse.layer2.l2.weight
        model_fine.layer4.weight = model_coarse.layer3.weight '''

        ## todo: 9) update fine weights (maybe with line search)
        ## for now, without line search
        x2 = torch.sub(x1,e2,-lr)
        torch.nn.utils.vector_to_parameters(x2,model_fine.parameters())

        ## 10) iterate on fine level N4 times
        for i in range(N4):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer_fine.zero_grad()
            loss.backward()
            optimizer_fine.step()

        if batch % 100 == 0:
            loss, current = loss_fn_fine.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #print('weights  w1 in training loop of first residual block: ', model.layer1.l1.weight)  # state_dict()['weight'])



## classical training (minibatch sgd)
def train_classical(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #model.layer1.l1.weight = torch.nn.parameter.Parameter(data=torch.ones(10,10))
            #model.layer1.l1.weight = model.layer2.l1.weight
            #print('weights  w1 in training loop of first residual block: ', model.state_dict().keys())


# can stay
def test(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 1 #2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_classical(train_dataloader, model, loss_fn, optimizer_fine)
    test(test_dataloader, model, loss_fn)
print('Now we look at state_dict.')
for key in model.state_dict():
    print(key)
    #print(model.state_dict()[key])
#print('weights  w1 in training loop of first residual block: ', model.state_dict().keys())
grads = []
paras = []
print('model.parameters',torch.nn.utils.parameters_to_vector(model.parameters()).size())
for param in model.parameters():
    grads.append(param.grad.view(-1))
    #print(param.size())
    #print(param)
    paras.append(param.flatten())
grads = torch.cat(grads)
paras = torch.cat(paras)
#print(grads.shape)
#print(paras.shape)
#print(grads)
print("Done!")




def train_2level_old(dataloader, model_fine, model_coarse, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model_fine.train()
    model_coarse.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Nested iteration (N1 iterations)
        ## iterate N1 times on coarse grid
        for i in range(N1):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_coarse(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ## prolongate to fine grid
        model_fine.layer0.weight = model_coarse.layer0.weight
        model_fine.layer1.l1.weight = model_coarse.layer1.l1.weight
        model_fine.layer1.l1.bias = model_coarse.layer1.l1.bias
        model_fine.layer1.l2.weight = model_coarse.layer1.l2.weight
        s = model_fine.layer2.l1.size()[0]
        model_fine.layer2.l1.weight = torch.nn.parameter.Parameter(data=torch.zeros(s, s))
        model_fine.layer2.l1.bias = torch.nn.parameter.Parameter(data=torch.zeros(s))
        model_fine.layer.l2.weight = torch.nn.parameter.Parameter(data=torch.zeros(s, s))
        model_fine.layer3.l1.weight = model_coarse.layer2.l1.weight
        model_fine.layer3.l1.bias = model_coarse.layer2.l1.bias
        model_fine.layer3.l2.weight = model_coarse.layer2.l2.weight
        model_fine.layer4.weight = model_coarse.layer3.weight

        #start V-cycle
        ##0) iterate on fine level N2 times
        for i in range(N2):
            # Compute prediction error
            pred = model_fine(X)
            loss = loss_fn_fine(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ###1) compute gradient on fine level #and restrict to coarse level
        g = []
        for param in model_fine.parameters():
            g.append(param.grad.view(-1))
        g = torch.cat(g)
        ###4) restrict gradient to coarse level
        g2 = torch.mm(restriction_matrix(10,2,sparse=False),g)

        ##2) restrict parameters to coarser level
        model_coarse.layer0.weight = model_fine.layer0.weight
        model_coarse.layer1.l1.weight = model_fine.layer1.l1.weight
        model_coarse.layer1.l1.bias = model_fine.layer1.l1.bias
        model_coarse.layer1.l2.weight = model_fine.layer1.l2.weight
        model_coarse.layer2.l1.weight = model_fine.layer3.l1.weight
        model_coarse.layer2.l1.bias = model_fine.layer3.l1.bias
        model_coarse.layer2.l2.weight = model_fine.layer3.l2.weight
        model_coarse.layer3.weight = model_fine.layer4.weight

        ###3) compute gradient of loss function on the coarse level
        g1 = []
        for param in model_coarse.parameters():
            g1.append(param.grad.view(-1))
        g1 = torch.cat(g1)

        ## 6) get new objective (instead of loss) function
        ### construct additional summand of new objective
        paras_flat = []
        for param in model.parameters():
            print(param.size)
            paras_flat.append(param.flatten())
        paras_flat = torch.cat(paras_flat)
        print(paras_flat.shape)

        nu = torch.sub(g2,g1)
        #new loss = old loss + torch.dot(nu,paras_flat) # how do we need to define a loss function?
        loss_fn_coarse_mod = loss_fn_coarse

        ##7) iterate on coarse level N3 times (with new objectives)
        for i in range(N3):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_coarse_mod(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### todo 8) compute prolongation of difference


        ''''## prolongate the parameters to the finer level (not needed)
        model_fine.layer0.weight = model_coarse.layer0.weight
        model_fine.layer1.l1.weight = model_coarse.layer1.l1.weight
        model_fine.layer1.l1.bias = model_coarse.layer1.l1.bias
        model_fine.layer1.l2.weight = model_coarse.layer1.l2.weight
        s = model_fine.layer2.l1.size()[0]
        model_fine.layer2.l1.weight = torch.nn.parameter.Parameter(data=torch.zeros(s,s))
        model_fine.layer2.l1.bias = torch.nn.parameter.Parameter(data=torch.zeros(s))
        model_fine.layer.l2.weight = torch.nn.parameter.Parameter(data=torch.zeros(s,s))
        model_fine.layer3.l1.weight = model_coarse.layer2.l1.weight
        model_fine.layer3.l1.bias = model_coarse.layer2.l1.bias
        model_fine.layer3.l2.weight = model_coarse.layer2.l2.weight
        model_fine.layer4.weight = model_coarse.layer3.weight '''

        ## todo: 9) update fine weights (maybe with line search)

        ## 10) iterate on fine level N4 times
        for i in range(N4):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print('weights  w1 in training loop of first residual block: ', torch.nn.utils.parameters_to_vector(model.parameters()))  # state_dict()['weight'])

#in model.parameter schleife: zuweisung von parameterwerten so möglich parameter.data = torch.zeros(8570)