"""
Implements pytorch autograd and nn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

from typing import OrderedDict
import torch
import torch.nn as nn
from a4_helper import *
import torch.nn.functional as F
import torch.optim as optim

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  
  print('Hello from pytorch_autograd_and_nn.py!!')



################################################################################
# Part II. Barebones PyTorch                         
################################################################################
# Before we start, we define the flatten function for your convenience.
def flatten(x, start_dim=1, end_dim=-1):
  return x.flatten(start_dim=start_dim, end_dim=end_dim)


def three_layer_convnet(x, params):
  """
  Performs the forward pass of a three-layer convolutional network with the
  architecture defined above.

  Inputs:
  - x: A PyTorch Tensor of shape (N, C, H, W) giving a minibatch of images
  - params: A list of PyTorch Tensors giving the weights and biases for the
    network; should contain the following:
    - conv_w1: PyTorch Tensor of shape (channel_1, C, KH1, KW1) giving weights
      for the first convolutional layer
    - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
      convolutional layer
    - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
      weights for the second convolutional layer
    - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
      convolutional layer
    - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
      figure out what the shape should be?
    - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
      figure out what the shape should be?
  
  Returns:
  - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
  """
  conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
  scores = None
  ##############################################################################
  # TODO: Implement the forward pass for the three-layer ConvNet.              
  # The network have the following architecture:                               
  # 1. Conv layer (with bias) with 32 5x5 filters, with zero-padding of 2     
  #   2. ReLU                                                                  
  # 3. Conv layer (with bias) with 16 3x3 filters, with zero-padding of 1     
  # 4. ReLU                                                                   
  # 5. Fully-connected layer (with bias) to compute scores for 10 classes    
  # Hint: F.linear, F.conv2d, F.relu, flatten (implemented above)                                   
  ##############################################################################
  x = torch.nn.functional.relu(torch.nn.functional.conv2d(x, conv_w1, conv_b1, padding=2))
  x = torch.nn.functional.relu(torch.nn.functional.conv2d(x, conv_w2, conv_b2, padding=1))
  x = flatten(x)
  scores = torch.nn.functional.linear(x, fc_w, fc_b)
  ##############################################################################
  #                                 END OF YOUR CODE                             
  ##############################################################################
  return scores


def initialize_three_layer_conv_part2(dtype=torch.float, device='cpu'):
  '''
  Initializes weights for the three_layer_convnet for part II
  Inputs:
    - dtype: A torch data type object; all computations will be performed using
        this datatype. float is faster but less accurate, so you should use
        double for numeric gradient checking.
      - device: device to use for computation. 'cpu' or 'cuda'
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  kernel_size_2 = 3

  # Initialize the weights
  conv_w1 = None
  conv_b1 = None
  conv_w2 = None
  conv_b2 = None
  fc_w = None
  fc_b = None

  ##############################################################################
  # TODO: Define and initialize the parameters of a three-layer ConvNet           
  # using nn.init.kaiming_normal_. You should initialize your bias vectors    
  # using the zero_weight function.                         
  # You are given all the necessary variables above for initializing weights. 
  ##############################################################################
  shape1 = (channel_1, C, kernel_size_1, kernel_size_1)
  conv_w1 = torch.nn.init.kaiming_normal_(torch.empty(shape1, dtype=dtype, device=device))
  conv_b1 = torch.nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device=device))
  conv_w1.requires_grad = True
  conv_b1.requires_grad = True
  H_1 = (H - kernel_size_1 + 2*2)//1 +1   #conv1 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
  W_1 = (W - kernel_size_1 + 2*2)//1 +1
  
  shape2 = (channel_2, channel_1, kernel_size_2, kernel_size_2)
  conv_w2 = torch.nn.init.kaiming_normal_(torch.empty(shape2, dtype=dtype, device=device))
  conv_b2 = torch.nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device=device))
  conv_w2.requires_grad = True
  conv_b2.requires_grad = True
  H_2 = (H_1 - kernel_size_2 + 2*1)//1 +1   #conv2 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
  W_2 = (W_1 - kernel_size_2 + 2*1)//1 +1
  
  shape3_fc = (num_classes, channel_2 * H_2 * W_2)
  fc_w = torch.nn.init.kaiming_normal_(torch.empty(shape3_fc, dtype=dtype, device=device))
  fc_b = torch.nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device=device))
  fc_w.requires_grad = True
  fc_b.requires_grad = True

  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]




################################################################################
# Part III. PyTorch Module API                         
################################################################################

class ThreeLayerConvNet(nn.Module):
  def __init__(self, in_channel, channel_1, channel_2, num_classes):
    super().__init__()
    ############################################################################
    # TODO: Set up the layers you need for a three-layer ConvNet with the       
    # architecture defined below. You should initialize the weight  of the
    # model using Kaiming normal initialization, and zero out the bias vectors.     
    #                                       
    # The network architecture should be the same as in Part II:          
    #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2  
    #   2. ReLU                                   
    #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
    #   4. ReLU                                   
    #   5. Fully-connected layer to num_classes classes               
    #                                       
    # We assume that the size of the input of this network is `H = W = 32`, and   
    # there is no pooing; this information is required when computing the number  
    # of input channels in the last fully-connected layer.              
    #                                         
    # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_            
    ############################################################################
    H = W = 32
    self.conv1 = torch.nn.Conv2d(in_channel, channel_1, kernel_size=(5,5), padding_mode="zeros", padding=2)
    H_1 = (H - 5 + 2*2)//1 +1   #conv1 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
    W_1 = (W - 5 + 2*2)//1 +1   
    self.conv2 = torch.nn.Conv2d(channel_1, channel_2, (3,3), padding=1) #default zero padding
    H_2 = (H_1 - 3 + 2*1)//1 +1   #conv1 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
    W_2 = (W_1 - 3 + 2*1)//1 +1
    self.fc1 = torch.nn.Linear(channel_2 * H_2 * W_2, num_classes)

    torch.nn.init.kaiming_normal_(self.conv1.weight)
    torch.nn.init.zeros_(self.conv1.bias)
    torch.nn.init.kaiming_normal_(self.conv2.weight)
    torch.nn.init.zeros_(self.conv2.bias)
    torch.nn.init.kaiming_normal_(self.fc1.weight)
    torch.nn.init.zeros_(self.fc1.bias)

    ############################################################################
    #                           END OF YOUR CODE                            
    ############################################################################

  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function for a 3-layer ConvNet. you      
    # should use the layers you defined in __init__ and specify the       
    # connectivity of those layers in forward()   
    # Hint: flatten (implemented at the start of part II)                          
    ############################################################################
    x = torch.nn.functional.relu(self.conv1(x))
    x = torch.nn.functional.relu(self.conv2(x))
    x = flatten(x)
    scores = self.fc1(x)
    ############################################################################
    #                            END OF YOUR CODE                          
    ############################################################################
    return scores


def initialize_three_layer_conv_part3():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part III
  '''

  # Parameters for ThreeLayerConvNet
  C = 3
  num_classes = 10

  channel_1 = 32
  channel_2 = 16

  # Parameters for optimizer
  learning_rate = 3e-3
  weight_decay = 1e-4

  model = None
  optimizer = None
  ##############################################################################
  # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.     
  # Use the above mentioned variables for setting the parameters.                
  # You should train the model using stochastic gradient descent without       
  # momentum, with L2 weight decay of 1e-4.                    
  ##############################################################################
  model = ThreeLayerConvNet(3, channel_1, channel_2, num_classes)
  optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, weight_decay= weight_decay)
  ##############################################################################
  #                                 END OF YOUR CODE                            
  ##############################################################################
  return model, optimizer


################################################################################
# Part IV. PyTorch Sequential API                        
################################################################################

# Before we start, We need to wrap `flatten` function in a module in order to stack it in `nn.Sequential`.  #!!!! önemli !!!!
# As of 1.3.0, PyTorch supports `nn.Flatten`, so this is not required in the latest version.
# However, let's use the following `Flatten` class for backward compatibility for now.
class Flatten(nn.Module):
  def forward(self, x):
    return flatten(x)


def initialize_three_layer_conv_part4():
  '''
  Instantiates a ThreeLayerConvNet model and a corresponding optimizer for part IV
  '''
  # Input/Output dimenssions
  C, H, W = 3, 32, 32
  num_classes = 10

  # Hidden layer channel and kernel sizes
  channel_1 = 32
  channel_2 = 16
  kernel_size_1 = 5
  pad_size_1 = 2
  kernel_size_2 = 3
  pad_size_2 = 1

  # Parameters for optimizer
  learning_rate = 1e-2
  weight_decay = 1e-4
  momentum = 0.5

  model = None
  optimizer = None
  ##################################################################################
  # TODO: Rewrite the 3-layer ConvNet with bias from Part III with Sequential API and 
  # a corresponding optimizer.
  # You don't have to re-initialize your weight matrices and bias vectors.  
  # Here you should use `nn.Sequential` to define a three-layer ConvNet with:
  #   1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2 
  #   2. ReLU                                      
  #   3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1 
  #   4. ReLU                                      
  #   5. Fully-connected layer (with bias) to compute scores for 10 classes        
  #                                            
  # You should optimize your model using stochastic gradient descent with Nesterov   
  # momentum 0.5, with L2 weight decay of 1e-4 as given in the variables above.   
  # Hint: nn.Sequential, Flatten (implemented at the start of Part IV)   
  ####################################################################################
  H_1 = (H - kernel_size_1 + 2*pad_size_1)//1 +1   #conv1 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
  W_1 = (W - 5 + 2*2)//1 +1
  H_2 = (H_1 - 3 + 2*1)//1 +1   #conv1 layer sonrası H',W' hesaplıyoruz : H’ = ( (H – K + 2P) / S ) + 1
  W_2 = (W_1 - 3 + 2*1)//1 +1
  
  
  model = nn.Sequential(OrderedDict([
                        ("conv1",torch.nn.Conv2d(C, channel_1, (kernel_size_1, kernel_size_1), padding=pad_size_1, bias=True)),
                        ("relu1",torch.nn.ReLU()),
                        ("conv2",torch.nn.Conv2d(channel_1, channel_2, (kernel_size_2, kernel_size_2), padding= pad_size_2, bias=True)),
                        ("relu2",torch.nn.ReLU()),
                        ("flatten",Flatten()),
                        ("fc1",torch.nn.Linear(H_2 * W_2 * channel_2, num_classes, bias=True))
                        ]))

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
  ################################################################################
  #                                 END OF YOUR CODE                             
  ################################################################################
  return model, optimizer


################################################################################
# Part V. ResNet for CIFAR-10                        
################################################################################

class PlainBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()
    
    self.net = None
    ############################################################################
    # TODO: Implement PlainBlock.                                             
    # Hint: Wrap your layers by nn.Sequential() to output a single module.     
    #       You don't have use OrderedDict.                                    
    # Inputs:                                                                  
    # - Cin: number of input channels                                          
    # - Cout: number of output channels                                        
    # - downsample: add downsampling (a conv with stride=2) if True            
    # Store the result in self.net.                                            
    ############################################################################
    s = 2 if downsample else 1  #true ise 2 false ise 1 return edicek
    Cmid = (Cout + Cin) // 2    
    self.net = torch.nn.Sequential(torch.nn.BatchNorm2d(Cin),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(Cin, Cmid, kernel_size=(3,3), stride=s, padding=1),
                        torch.nn.BatchNorm2d(Cmid),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(Cmid, Cout, kernel_size=(3,3), stride=1, padding=1)
                        )
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.net(x)


class ResidualBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None # F
    self.shortcut = None # G
    ############################################################################
    # TODO: Implement residual block using plain block. Hint: nn.Identity()    #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    self.block = PlainBlock(Cin, Cout, downsample)
    if Cin == Cout and downsample == False:                            #[1.durum]: F(x) derinlik ve uzamsal size değişikliği yapmıyor ise
      self.shortcut = torch.nn.Identity()
    elif downsample == False:                                          #[2.durum]: F(x) uzamsal olarak değil ancak derinlikte değişiklik yapıyor ise
      self.shortcut = torch.nn.Conv2d(Cin, Cout, (1,1), stride=1)      #1x1 conv çözünürlüğü değiştirmeyeceğinden paddig'e vs gerek yok
    elif Cin != Cout and downsample != False:
      self.shortcut = torch.nn.Conv2d(Cin, Cout, (1,1), stride=2)      #[3.durum]: F(x) hem uzamsal hemde derinlik olarak değişiklik yaptığında
    else:                                                              #1x1 conv'da stride ayarlayarak çözünürlük/uzamsallıkta da değişiklik yapılabiliyor!!!
      print("Unexpected Condition!!!!")                                #yani sadece derinlik değiştirme yapmayabilir

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
  
  def forward(self, x):
    return self.block(x) + self.shortcut(x)   # R(x) = F(x) + G(x)


#############################################################################################################################################
class ResNet(nn.Module):
  def __init__(self, stage_args, Cin=3, block=ResidualBlock, num_classes=10):
    super().__init__()

    self.cnn = None
    ############################################################################
    # TODO: Implement the convolutional part of ResNet using ResNetStem,       #
    #       ResNetStage, and wrap the modules by nn.Sequential.                #
    # Store the model in self.cnn.                                             #
    ############################################################################
    layers = [ResNetStem(Cin, stage_args[0][0])]

    for stage_arg in stage_args:                                                
      layers.append(ResNetStage(*stage_arg))
    
    self.cnn = torch.nn.Sequential(*layers)
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    self.fc = nn.Linear(stage_args[-1][1], num_classes)
  
  def forward(self, x):
    scores = None
    ############################################################################
    # TODO: Implement the forward function of ResNet.                          #
    # Store the output in `scores`.                                            #
    ############################################################################
    x = self.cnn(x)
    x = torch.nn.AvgPool2d((x.shape[-2],x.shape[-1]))(x)                        #x.shape:(NXCXHXW) böylece-> x.shape[-2]:H,x.shape[-1]:W
    x = flatten(x)
    scores = self.fc(x)
    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################
    return scores


class ResidualBottleneckBlock(nn.Module):
  def __init__(self, Cin, Cout, downsample=False):
    super().__init__()

    self.block = None
    self.shortcut = None
    ############################################################################
    # TODO: Implement residual bottleneck block.                               #
    # Inputs:                                                                  #
    # - Cin: number of input channels                                          #
    # - Cout: number of output channels                                        #
    # - downsample: add downsampling (a conv with stride=2) if True            #
    # Store the main block in self.block and the shortcut in self.shortcut.    #
    ############################################################################
    stride = 2 if downsample else 1
    self.block = torch.nn.Sequential(
                                    torch.nn.BatchNorm2d(Cin),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(Cin, Cout//4, kernel_size=(1,1), stride=stride),
                                    torch.nn.BatchNorm2d(Cout//4),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(Cout//4, Cout//4, kernel_size=(3,3), padding=(1,1), stride=1),
                                    torch.nn.BatchNorm2d(Cout//4),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(Cout//4, Cout, kernel_size=(1,1))
                                    )
    if Cin == Cout and downsample == False:
      self.shortcut = torch.nn.Identity()
    elif downsample == False:
      self.shortcut = torch.nn.Conv2d(Cin, Cout, (1,1), 1)
    elif Cin != Cout and downsample != False:
      self.shortcut = torch.nn.Conv2d(Cin, Cout, (1,1), stride=stride)
    else:                                                              
      print("Unexpected Condition!!!!") 

    ############################################################################
    #                                 END OF YOUR CODE                         #
    ############################################################################

  def forward(self, x):
    return self.block(x) + self.shortcut(x)

##############################################################################
# No need to implement anything here                     
##############################################################################
class ResNetStem(nn.Module):
  def __init__(self, Cin=3, Cout=8):
    super().__init__()
    layers = [
        nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    ]
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)

class ResNetStage(nn.Module):
  def __init__(self, Cin, Cout, num_blocks, downsample=True,
               block=ResidualBlock):
    super().__init__()
    blocks = [block(Cin, Cout, downsample)]
    for _ in range(num_blocks - 1):
      blocks.append(block(Cout, Cout))
    self.net = nn.Sequential(*blocks)
  
  def forward(self, x):
    return self.net(x)