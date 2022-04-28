import time
from collections import OrderedDict
import torch
from torch import nn

from Nets import ResBlock1, restriction, prolongation

device = 'cpu'
torch.set_num_threads(4)

def make_resnet(dims: list,ResBlock, no_reslayers,h=1. ,act_fun = "ReLU"):
    if act_fun == "ReLU":
        AF = nn.ReLU()
    if act_fun == "Tanh":
        AF = nn.Tanh()
    if act_fun == "Sigmoid":
        AF = nn.Sigmoid()

    layer_dict = OrderedDict()
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

def train_multilevel(dataloader,loss_fns, optimizers, lr, iteration_numbers,no_levels,coarse_no_reslayers,dims,ResBlock,act_fun="ReLU", Print=False):
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

    model_list = gen_hierarch_models(no_levels, coarse_no_reslayers, dims, ResBlock, act_fun)
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

        # for schleife für levels
        for i in range(no_levels -1):
            # nested iteration
            current_no_reslayers = coarse_no_reslayers
            nested_iteration(iteration_numbers[0],current_no_reslayers,model_list[i],model_list[i+1],loss_fns[i],optimizers[i],dims)
            #v-cycle (with variying depth)
            starting_no_reslayers = #todo: how do we compute that?
            Vcycle(i+1,iteration_numbers[1:],model_list[0:i+2], optimizers[0:i+2], loss_fns[0:i+2],dims,starting_no_reslayers)
            coarse_no_reslayers = get_no_of_finer_reslayers(coarse_no_reslayers)

        if batch % 100 == 0:
            current = batch*len(X)
            #loss = loss_fn_fine.item()
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if Print:
                print('iteration no. '+str(current)+'of'+str(size)) #batch*len(X))
                print('loss',loss_fine.item())   #todo what is loss_fine?
            tic2 = time.perf_counter()
            if Print:
                print('needed time for this batch: ', tic2 - toc2)
    tic = time.perf_counter()
    if Print:
        print('needed time for one epoch: ', tic-toc)


