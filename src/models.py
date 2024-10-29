import torch
from torch import nn

class Model(nn.Module):
    '''
    Model for the prediction of the next static and dynamic grid maps
    given the current static and dynamic grid maps.

    Input:
    - input_static: Static grid map of the current frame
    - input_dynamic: Dynamic grid map of the current frame

    Output:
    - output_static: Static grid map of the next frame
    - output_dynamic: Dynamic grid map of the next frame
    - output_free: Free space grid map of the next frame

    At the end, the static, dinamic and free maps are passed through a softmax
    layer to get the probabilities of each cell being in each class.
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, input_static, input_dynamic):
        # Concatenate the input static and dynamic maps along the channel dimension
        x = torch.cat((input_static, input_dynamic), dim=1)

        # Pass the input through the network
        x = self.nn(x)

        # Split the output into the static, dynamic and free maps
        output = x[:, 0:2, :, :]
        return output
    
class FlatCNN(nn.Module):
    '''
    Model for the prediction of the next static and dynamic grid maps
    given the current static and dynamic grid maps.

    The model has 4 convolutional layers with ReLU activation functions.

    At the end, the static, dinamic and free maps are passed through a softmax
    layer to get the probabilities of each cell being in each class.
    '''

    def __init__(self):
        super(FlatCNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, input_static, input_dynamic):
        # Pass the input through the network
        x = self.nn(x)

        # Split the output into the static, dynamic and free maps
        output = x[:, 0:2, :, :]
        return output
