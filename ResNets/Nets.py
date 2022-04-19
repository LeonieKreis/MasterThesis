import torch
from torch import nn
#from torchsummary import summary



#ResBlock1 is a residual block with two weights and one bias
class ResBlock1(nn.Module):
    def __init__(self, in_channels, out_channels,h):
        super().__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.shortcut = nn.Sequential()
        self.l2 = nn.Linear(out_channels, out_channels, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input,h=1):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.l1(input))
        input = self.l2(input)
        input = h*input + shortcut
        return input #or do we need nn.ReLU()(input)? ->look up



# a first personalized ResNet with variable h
class ResNet1(nn.Module):
    def __init__(self, in_channels, out_channels, resblock, h=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # we use h
        self.layer1 = nn.Sequential(
            resblock(28*28, 28*28, h)
        )

        self.layer2 = nn.Sequential(
            resblock(28*28, 28*28, h)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False)
        )


    def forward(self, input):
        input = self.flatten(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)

        return input


class ResNet1_coarse(nn.Module):
    def __init__(self, in_channels,reslayer_size, out_channels, resblock, h=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # we use h
        self.layer0 = nn.Linear(in_channels, reslayer_size, bias=False)
        self.layer1 = resblock(reslayer_size, reslayer_size, h)
        self.layer2 = resblock(reslayer_size, reslayer_size, h)
        self.layer3 = nn.Linear(reslayer_size, out_channels, bias=False)


    def forward(self, input):
        input = self.flatten(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)

        return input


class ResNet1_fine(nn.Module):
    def __init__(self, in_channels,reslayer_size, out_channels, resblock, h=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # we use h
        self.layer0 = nn.Linear(in_channels, reslayer_size, bias= False)
        self.layer1 = resblock(reslayer_size, reslayer_size, h)
        self.layer2 = resblock(reslayer_size, reslayer_size, h) # new layer
        self.layer3 = resblock(reslayer_size, reslayer_size, h)
        self.layer4 = nn.Linear(reslayer_size, out_channels, bias=False)



    def forward(self, input):
        input = self.flatten(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        return input

class ResNet1_fine2(nn.Module):
    def __init__(self, in_channels,reslayer_size, out_channels, resblock, h=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # we use h
        self.layer0 = nn.Linear(in_channels, reslayer_size, bias= False)
        self.layer1 = resblock(reslayer_size, reslayer_size, h)
        self.layer2 = resblock(reslayer_size, reslayer_size, h)  #new layer
        self.layer3 = resblock(reslayer_size, reslayer_size, h)
        self.layer4 = resblock(reslayer_size, reslayer_size, h)  #new layer
        self.layer5 = resblock(reslayer_size, reslayer_size, h)
        self.layer6 = nn.Linear(reslayer_size, out_channels, bias=False)



    def forward(self, input):
        input = self.flatten(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)

        return input




def prolongation(flat_parameter_tensor, reslayer_size, no_reslayers_coarse, dim_in, dim_out): # from coarse to fine grid
    # the discretization correction of the W_2 is not implemented yet
    # w_2finee = 2*w_2coarse
    dim_resblock = 2*reslayer_size*reslayer_size+reslayer_size
    no_reslayers_fine = 2*no_reslayers_coarse -1
    Q1flat, Res_coarse_flat, Q2flat = torch.split(flat_parameter_tensor, [dim_in*reslayer_size,no_reslayers_coarse*dim_resblock,reslayer_size*dim_out])
    #in each resblock, the last reslayer_size^2 parameters must be corrected with multiplicative factor 2
    t = torch.cat((Q1flat,Res_coarse_flat[0:dim_resblock]))
    for i in range(1,no_reslayers_coarse):
        t = torch.cat((t,torch.zeros(dim_resblock)))
        t2 = Res_coarse_flat[i*dim_resblock:(i+1)*dim_resblock]
        t = torch.cat((t,t2))
    t = torch.cat((t,Q2flat))
    return t

def restriction(flat_parameter_tensor, reslayer_size, no_reslayers_fine, dim_in, dim_out): # the discretization correction of the W_2 is not implemented yet
    #w_2coarse = 0.5*w_2fine
    # in each resblock, the last reslayer_size^2 parameters must be corrected with multiplicative factor 0.5
    dim_resblock = 2 * reslayer_size * reslayer_size + reslayer_size
    no_reslayers_coarse = int((no_reslayers_fine + 1)/2)
    Q1flat, Res_fine_flat, Q2flat = torch.split(flat_parameter_tensor,
                                                  [dim_in * reslayer_size, no_reslayers_fine * dim_resblock,
                                                   reslayer_size * dim_out])
    t = torch.cat((Q1flat,Res_fine_flat[0:dim_resblock]))
    for i in range(1,no_reslayers_fine):
        if i%2 ==0: #i even
            t2 = Res_fine_flat[i*dim_resblock:(i+1)*dim_resblock]
            t = torch.cat((t,t2))
        #else: # i uneven
            #do nothing, these resblocks will be cut out
    t = torch.cat((t,Q2flat))
    return t


dim_in = 28*28
dim_out = 10
reslayer_size = 10
dim_resblock = 2*reslayer_size*reslayer_size+reslayer_size
no_reslayers= int(2)
flat_parameter_tensor = torch.ones(28*28*reslayer_size+no_reslayers*dim_resblock+reslayer_size*10)
#print(flat_parameter_tensor.size())
p = prolongation(flat_parameter_tensor,reslayer_size,no_reslayers,dim_in,dim_out)
#print(p.size())

# as above, only in matrix form
def prolongation_matrix( reslayer_size, no_reslayers,dim_in, dim_out, sparse=True):
    dim_resblock = int(2 * reslayer_size * reslayer_size + reslayer_size)
    no_reslayers_fine = int(2 * no_reslayers - 1)
    dimQ1=dim_in*reslayer_size
    dimQ2=reslayer_size*dim_out
    P = torch.cat((torch.eye(dimQ1), torch.zeros(dimQ1,int(no_reslayers*dim_resblock+dimQ2))),1)
    for i in range(no_reslayers):
        rb = torch.zeros(dim_resblock, dimQ1)
        z = torch.zeros(dim_resblock,no_reslayers*dim_resblock+dimQ2+dimQ1)
        for j in range(no_reslayers):
            if i==j:
                rb = torch.cat((rb, torch.eye(dim_resblock)), 1)
            else:
                rb = torch.cat((rb,torch.zeros(dim_resblock,dim_resblock)),1)
        rb = torch.cat((rb, torch.zeros(dim_resblock,dimQ2)),1)
        if i != no_reslayers-1:
            P = torch.cat((P, rb, z), 0)
        else:
            P = torch.cat((P, rb), 0)

    last_row = torch.cat((torch.zeros(dimQ2,dimQ1+no_reslayers*dim_resblock),torch.eye(dimQ2)),1)
    P = torch.cat((P, last_row),0)
    if sparse:
        P.to_sparse()
    return P

P = prolongation_matrix(reslayer_size, no_reslayers,dim_in, dim_out, sparse=False)
#print('size of prolongation matrix',P.size())

def restriction_matrix( reslayer_size, no_reslayers,dim_in,dim_out, sparse=True):
    P = prolongation_matrix(reslayer_size,no_reslayers,dim_in,dim_out,sparse=False)
    Q = torch.t(P)
    if sparse:
        Q = Q.to_sparse()
    return Q

Q = restriction_matrix(reslayer_size, no_reslayers,dim_in, dim_out, sparse=True)
#print('size of restriction matrix',Q.size())


#ResBlock2 is a residual block with one weights and one bias
class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, h):
        super().__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.shortcut = nn.Sequential()
        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input,h=1):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.l1(input))
        input = h*input + shortcut
        return input #or do we need nn.ReLU()(input)? ->look up



# a first personalized ResNet with variable h
class ResNet2(nn.Module):
    def __init__(self, in_channels,out_channels, resblock, h=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # we use h
        self.layer1 = nn.Sequential(
            resblock(28*28, 28*28, h)
        )

        self.layer2 = nn.Sequential(
            resblock(28*28, 28*28, h)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False)
        )


    def forward(self, input):
        input = self.flatten(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)

        return input


