import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evotorch.neuroevolution import NEProblem


# Neural network
class NeuralNetwork(nn.Module):
    """
    3 layer neural network:
        - Input layer: 24 nodes
        - Hidden layer: 20 nodes
        - Output layer: 4 nodes

        - Activation function: Tanh
    """

    def __init__(self, state_size, hidden_size, action_size):
        super(NeuralNetwork, self).__init__()  # Call the parent class constructor
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_size)
        
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()

    
    def forward(self, state):
        state = torch.Tensor(state)  # Convert state to a Tensor
        x = self.act1(self.layer1(state))
        x = self.act2(self.layer2(x))
        return x


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")
    
    

def parameterize_net(network, parameters: torch.Tensor, network_device="cpu") -> nn.Module:
    """Parameterize the network with a given set of parameters.
    Args:
        parameters (torch.Tensor): The parameters with which to instantiate the network
    Returns:
        instantiated_network (nn.Module): The network instantiated with the parameters
    """

    # Move the parameters if needed
    if parameters.device != network_device:
        parameters = parameters.to(network_device)

    # Fill the network with the parameters
    fill_parameters(network, parameters)

    # Return the network
    return network
    
