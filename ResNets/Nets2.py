import time
from collections import OrderedDict
import torch
from torch import nn

from Nets import ResBlock1, restriction, prolongation

device = 'cpu'
torch.set_num_threads(4)

## 2-level nested iteration and mu-cycle
def train_2level(dataloader, model_fine, model_coarse, loss_fn_fine, loss_fn_coarse, optimizer_fine, optimizer_coarse, lr, iteration_numbers,no_reslayers_coarse, prolong_matrix = False, Print=False):
    device = 'cpu'
    size = len(dataloader.dataset)
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

        if batch%100 == 0:
            test1 = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        # Nested iteration (N1 iterations)
        ## iterate N1 times on coarse grid
        N1 = iteration_numbers[0]
        for i in range(N1):
            # Compute prediction error
            pred = model_coarse(X)
            loss_coarse = loss_fn_coarse(pred, y)

            # Backpropagation
            optimizer_coarse.zero_grad()
            loss_coarse.backward()
            optimizer_coarse.step()

        if batch%100 == 0:
            test2 = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            if torch.equal(test1,test2):
                print('1) model parameters have not changed while nested iteration!')
        ## prolongate to fine grid
        vec_coarse = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        if prolong_matrix:
            vec_fine = torch.mv(prolong,vec_coarse)
        else:
            vec_fine = prolongation(vec_coarse, reslayer_size, no_reslayers_coarse, dim_in, dim_out)
        torch.nn.utils.vector_to_parameters(vec_fine,model_fine.parameters())

        #start V-cycle
        ##0) iterate on fine level N2 times
        N2 = iteration_numbers[1]
        for i in range(N2):
            # Compute prediction error
            pred = model_fine(X)
            loss_fine = loss_fn_fine(pred, y)

            # Backpropagation
            optimizer_fine.zero_grad()
            loss_fine.backward()
            optimizer_fine.step()
        x1 = torch.nn.utils.parameters_to_vector(model_fine.parameters())

        # again testing if model_corase param have changed
        if batch%100 == 0:
            test3 = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            if not torch.equal(test3,test2):
                print('2) model parameters have changed and shouldnt!')
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
        #x1_bar = vec_coarse
        torch.nn.utils.vector_to_parameters(vec_coarse, model_coarse.parameters())
        x1_bar = torch.nn.utils.parameters_to_vector(model_coarse.parameters())

        if batch%100 == 0:
            test4 = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            if torch.equal(test3,test4):
                print('3) model parameters have not changed after restriction from fine level!')

        ###3) compute gradient of loss function on the coarse level
        g1 = []
        for param in model_coarse.parameters():
            g1.append(param.grad.view(-1))
        g1 = torch.cat(g1)

        ## 6) get new objective (instead of loss) function
        ### construct additional summand of new objective
        def loss_fn_coarse_mod(pred,y):
            paras_flat = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            nu = torch.sub(g2, g1)
            # new loss = old loss + torch.dot(nu,paras_flat) # how do we need to define a loss function?
            loss = torch.sub(loss_fn_coarse(pred,y),torch.dot(nu,paras_flat))
            return loss

        ##7) iterate on coarse level N3 times (with new objectives)
        N3 = iteration_numbers[2]
        for i in range(N3):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_coarse_mod(pred, y)
            #if batch%100 == 0:
                ##checking whether the modified loss works
                #print('modified loss:', loss.item())
                #loss2 = loss_fn_coarse(pred,y)
                #print('CEloss: ',loss2.item())

            # Backpropagation
            optimizer_coarse.zero_grad()
            loss.backward()
            optimizer_coarse.step()

        if batch%100 == 0:
            test5 = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            if torch.equal(test5,test4):
                print('4) model parameters have not changed while iterating on bottom!')

        ### 8) compute prolongation of difference
        x2_bar = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        e2_bar = torch.sub(x2_bar,x1_bar)
        if prolong_matrix:
            e2 = torch.mv(prolong,e2_bar)
        else:
            e2 = prolongation(e2_bar,reslayer_size,no_reslayers_coarse,dim_in,dim_out)

        #if batch%100 == 0:
            #print('norm of gradient: ',torch.dot(g,g).item())
            #print('norm of e2_bar: ',torch.dot(e2_bar,e2_bar).item())
            #print('norm of e2: ', torch.dot(e2, e2).item())
            #check = torch.dot(e2,g)
            #print('value of gradientf(x1)Te_2: ',check.item())
        if torch.equal(x2_bar, x1_bar):
            print('x1bar is x2bar!!')
        # check whether e2 is a descent direction:
        check = torch.dot(e2, g)
        if check.item() >= 0:
            print('e2 is not a descent direction!, has value: ', check.item())
        ## 9) update fine weights (maybe with line search)
        ## for now, without line search
        x2 = torch.sub(x1,e2,alpha=-lr)
        torch.nn.utils.vector_to_parameters(x2,model_fine.parameters())

        ## 10) iterate on fine level N4 times
        N4 = iteration_numbers[3]
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
                print('iteration no. '+str(batch*len(X))+' of '+str(size)+' loss: '+str(loss_fine.item()))
            tic2 = time.perf_counter()
            if Print:
                print('batchtime: ', tic2 - toc2)
    tic = time.perf_counter()
    if Print:
        print('epochtime: ', tic-toc)



## classical training (minibatch sgd)
def train_classical(dataloader, model, loss_fn, optimizer, Print=False):
    device = 'cpu'
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
        x1_bar = torch.nn.utils.parameters_to_vector(model.parameters())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x2_bar = torch.nn.utils.parameters_to_vector(model.parameters())
        #check whether the parameters have changed...
        if batch%100 == 0:
            if torch.equal(x1_bar,x2_bar):
                print('parameters of model have not changed')
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if Print:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            tic2 = time.perf_counter()
            if Print:
                print('batchtime: ', tic2-toc2)
            #model.layer1.l1.weight = torch.nn.parameter.Parameter(data=torch.ones(10,10))
            #model.layer1.l1.weight = model.layer2.l1.weight
            #print('iteration no. ', batch*len(X))
    tic = time.perf_counter()
    if Print:
        print('epochtime: ', tic-toc)

# can stay
def test(dataloader, model, loss_fn, Print = False):
    device = 'cpu'
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




def make_resnet(dims: list,ResBlock, no_reslayers,h=1. ,act_fun = "ReLU"):
    if act_fun == "ReLU":
        AF = nn.ReLU()
    if act_fun == "Tanh":
        AF = nn.Tanh()
    if act_fun == "Sigmoid":
        AF = nn.Sigmoid()

    layer_dict = OrderedDict()
    layer_dict[f'flatten0'] = nn.Flatten()
    layer_dict[f'linear0'] = nn.Linear(in_features=dims[0], out_features=dims[1], bias=False)
    #layer_dict[f'actfun0'] = AF
    for k in range(1,no_reslayers+1):
        layer_dict[f'res{k}'] = ResBlock(dims[1],dims[1],h )#nn.Linear(in_features=dims[1], out_features=dims[1])
        # layer_dict[f'actfun{k}'] = AF
    k_end = no_reslayers+1
    layer_dict[f'linear{k_end}'] = nn.Linear(in_features=dims[1], out_features=dims[2], bias=False)
    #layer_dict[f'actfun{k_end}'] = AF
    return nn.Sequential(layer_dict)

model_res = make_resnet([28*28,100,10], ResBlock1, no_reslayers=5,h=0.25)
#print(model_res)

def gen_hierarch_models(no_levels,coarse_no_reslayers,dims,ResBlock, act_fun="ReLU"):
    no_reslayers = coarse_no_reslayers # we start to build the coarsest net first
    h = 1
    model_list = []
    for k in range(no_levels):
        model = make_resnet(dims,ResBlock,no_reslayers,h=h,act_fun=act_fun)
        # write model in list
        model_list.append(model)
        #save model
        PATH = "nets/model_atlevel_"+str(k)+".pt"
        torch.save(model.state_dict(),PATH)
        no_reslayers = 2*no_reslayers-1
        h = 0.5*h
    print("Models of all levels are also saved in directory 'nets'!")
    return model_list

model_list = gen_hierarch_models(3,2,[28*28,100,10],ResBlock1)
#print(model_list)

def get_no_of_finer_reslayers(no_res_coarse,steps=1):
    for i in range(steps):
        no_res_coarse = int(2*no_res_coarse-1)
    return no_res_coarse

def get_no_of_coarser_reslayers(no_res_fine, steps=1):
    for i in range(steps):
        no_res_fine = (no_res_fine+1)/2
    return int(no_res_fine)

def train_multilevel(dataloader,model_list, loss_fns, optimizers, lr, iteration_numbers,no_levels,coarse_no_reslayers,dims, Print=False):
    toc = time.perf_counter()
    def nested_iteration(no_iteration,current_no_reslayers, model_coarse, model_fine, loss_fn_coarse,optimizer_coarse, dims):
        dim_in = dims[0]
        reslayer_size = dims[1]
        dim_out = dims[2]
        no_reslayers_coarse = current_no_reslayers
        # Nested iteration (N1=no_iteration)
        ## iterate N1 times on coarse grid
        for i in range(no_iteration):
            # Compute prediction error
            pred = model_coarse(X)
            loss_coarse = loss_fn_coarse(pred, y)
            # Backpropagation
            optimizer_coarse.zero_grad()
            loss_coarse.backward()
            optimizer_coarse.step()

        ## prolongate to fine grid
        vec_coarse = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
        vec_fine = prolongation(vec_coarse, reslayer_size, no_reslayers_coarse, dim_in, dim_out)
        torch.nn.utils.vector_to_parameters(vec_fine, model_fine.parameters())
        #return vec_fine

    def Vcycle(depth,needed_itnumbers,models, optimizers, loss_fns,dims, starting_no_reslayers):
        dim_in = dims[0]
        reslayer_size = dims[1]
        dim_out = dims[2]

        no_reslayers_fine = starting_no_reslayers
        #models, optimizers and loss functions must go from coarse to fine!
        x1_list = [] # finest to coarstest level
        x1bar_list = []
        no_models = len(models)
        if no_models != depth+1:
            return print("error! provided models:"+str(no_models)+ "and depth of vcycle:"+str(depth)+" do not match!")
        #downwards
        for l in range(depth):
            #in l= 0:   modelcoarse=models[vorletzte]=[no_models-2]=[no_models-(l+2)]
            #           modelfine=model[letzte]=[no_models-1]=[no_models-(l+1)]
            model_coarse = models[no_models-(l+2)]
            model_fine = models[no_models-(l+1)]
            #optimizer_coarse = optimizers[no_models-(l+2)] not needed
            optimizer_fine = optimizers[no_models - (l + 1)]
            loss_fn_coarse = loss_fns[no_models-(l+2)]
            if l == 0:
                loss_fn_fine = loss_fns[no_models-(l+1)]


            #0) iterate on fine level
            N2 = needed_itnumbers[0]
            for i in range(N2):
                # Compute prediction error
                pred = model_fine(X)
                loss_fine = loss_fn_fine(pred, y)
                # Backpropagation
                optimizer_fine.zero_grad()
                loss_fine.backward()
                optimizer_fine.step()
            x1 = torch.nn.utils.parameters_to_vector(model_fine.parameters())
            #-> save x1 in list
            x1_list.append(x1)

            #1)compute gradient on fine level
            g = []
            for param in model_fine.parameters():
                g.append(param.grad.view(-1))
            g = torch.cat(g)

            #4) restrict gradient to coarse level
            g2 = restriction(g, reslayer_size, no_reslayers_fine, dim_in, dim_out)

            #2) restrict parameters to coarse level
            vec_fine = torch.nn.utils.parameters_to_vector(model_fine.parameters())
            vec_coarse = restriction(vec_fine, reslayer_size, no_reslayers_fine, dim_in, dim_out)
            x1_bar = vec_coarse
            torch.nn.utils.vector_to_parameters(vec_coarse, model_coarse.parameters())
            #->save x1bar in list
            x1bar_list.append(x1_bar)

            #3) compute coarse gradient
            g1 = []
            for param in model_coarse.parameters():
                g1.append(param.grad.view(-1))
            g1 = torch.cat(g1)

            #5) compute difference of gradients
            nu = torch.sub(g2, g1)

            #6) define new objective
            def loss_fn_coarse_mod(pred, y):
                paras_flat = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
                loss = torch.sub(loss_fn_coarse(pred, y), torch.dot(nu, paras_flat))
                return loss
            loss_fn_fine=loss_fn_coarse_mod # new objective function for next step in loop

            no_reslayers_fine = get_no_of_coarser_reslayers(no_reslayers_fine)
            #end for loop with l

        #at bottom
        # model_coarse = models[0]
        optimizer_coarse = optimizers[0]
        N3 = needed_itnumbers[1]
        for i in range(N3):
            # Compute prediction error
            pred = model_coarse(X)
            loss = loss_fn_fine(pred, y)
            # Backpropagation
            optimizer_coarse.zero_grad()
            loss.backward()
            optimizer_coarse.step()

        #upwards
        no_reslayers_coarse = no_reslayers_fine
        for l in range(depth):
            model_coarse = models[l]
            model_fine = models[l+1]
            optimizer_fine = optimizers[l+1]

            #8) compute prolongation of difference
            x2_bar = torch.nn.utils.parameters_to_vector(model_coarse.parameters())
            e2_bar = torch.sub(x2_bar, x1bar_list[depth -l -1])
            e2 = prolongation(e2_bar, reslayer_size, no_reslayers_coarse, dim_in, dim_out)

            #9) update fine weights (maybe with line search)
            ## for now, without line search
            x2 = torch.sub(x1_list[depth -l -1], e2, alpha=-lr)
            torch.nn.utils.vector_to_parameters(x2, model_fine.parameters())

            ## 10) iterate on fine level
            N4 = needed_itnumbers[2]
            for i in range(N4):
                # Compute prediction error
                pred = model_fine(X)
                loss_fine = loss_fn_fine(pred, y)
                # Backpropagation
                optimizer_fine.zero_grad()
                loss_fine.backward()
                optimizer_fine.step()
            no_reslayers_coarse = get_no_of_finer_reslayers(no_reslayers_coarse)
        #return 0

    # wir brauchen auch vermutlich eine list von optimierern und loss functions for each level
    size = len(dataloader.dataset)
    for i in range(no_levels):
        #PATH = "nets/model_atlevel_"+str(k)+".pt"
        #model.load_state_dict(torch.load(PATH))
        #model = model_list[i]
        model_list[i].train()
    #models.train
    for batch, (X,y) in enumerate(dataloader):
        if batch%100 == 0:
            toc2 = time.perf_counter()
        X,y = X.to(device),y.to(device)

        starting_no_reslayers = get_no_of_finer_reslayers(coarse_no_reslayers)
        # for schleife für levels
        for i in range(no_levels -1):
            # nested iteration
            current_no_reslayers = coarse_no_reslayers
            nested_iteration(iteration_numbers[0],current_no_reslayers,model_list[i],model_list[i+1],loss_fns[i],optimizers[i],dims)
            #v-cycle (with variying depth)
            Vcycle(i+1,iteration_numbers[1:],model_list[0:i+2], optimizers[0:i+2], loss_fns[0:i+2],dims,starting_no_reslayers)
            starting_no_reslayers =  get_no_of_finer_reslayers(starting_no_reslayers)
            coarse_no_reslayers = get_no_of_finer_reslayers(coarse_no_reslayers)

        if batch % 100 == 0:
            current = batch*len(X)
            #loss = loss_fn_fine.item()
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if Print:
                print('iteration no. '+str(current)+' of '+str(size)+'  loss',loss_fns[len(loss_fns)-1](model_list[len(loss_fns)-1](X),y).item())
            tic2 = time.perf_counter()
            if Print:
                print('batchtime: ', tic2 - toc2)
    tic = time.perf_counter()
    if Print:
        print('epochtime: ', tic-toc)


'''before multilevel trianing: generate needed models via: model_list = gen_hierarch_models(no_levels, coarse_no_reslayers, dims, ResBlock, act_fun)'''