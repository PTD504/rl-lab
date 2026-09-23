import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_channels, height, output_dim):
        """
        Build a simple convolutional neural network with three convolutional layers to play the atari games
        """
        super(NeuralNetwork, self).__init__()
        # Calculate the size of the output from the convolutional layers, assume that the input is a square image
        conv1_output_height = (height - 8) // 4 + 1
        conv2_output_height = (conv1_output_height - 4) // 2 + 1
        conv3_output_height = (conv2_output_height - 3) + 1
        conv_output_size = conv3_output_height * conv3_output_height * 64

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, state):
        """
        Calculate the Q-values for a given state
        """
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values