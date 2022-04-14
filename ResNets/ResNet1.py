import torch
import numpy as np
from torch import nn
#from torchsummary import summary

from Nets import ResBlock1,  ResNet1, ResBlock2, ResNet2

resnet1 = ResNet1(28*28,10, ResBlock1, h=0.1)
resnet11 = ResNet1(28*28,10, ResBlock1, h=1)
#resnet1.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
resnet1.to("cpu")
resnet11.to("cpu")
#summary(resnet1, (3, 224, 224))
resnet1.eval()
print(resnet1)

#print('parameter list',nn.ParameterList())

resnet2 = ResNet2(28*28,10, ResBlock2, h=0.1)
resnet22 = ResNet2(28*28,10, ResBlock2, h=1)
#resnet1.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
resnet2.to("cpu")
resnet22.to("cpu")
#summary(resnet1, (3, 224, 224))
resnet2.eval()
print(resnet2)

#print(resnet1(torch.tensor(np.ones(64)).float()))
#print(resnet2(torch.tensor(np.ones(64)).float()))

## useful code:

#get gradient of parametrs of the model
'''model = models.resnet50()
# Calculate dummy gradients
model(torch.randn(1, 3, 224, 224)).mean().backward()
grads = []
for param in model.parameters():
    grads.append(param.grad.view(-1))
grads = torch.cat(grads)
print(grads.shape)
> torch.Size([25557032])'''