

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    A Long Short-Term Memory (LSTM) network for time-series forecasting.
    
    This model takes a sequence of historical data and outputs a prediction for 
    a future time step.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        """
        Initializes the LSTMForecaster model.

        Args:
            input_dim (int): The number of input features for each time step. 
                             For univariate stock prices, this is typically 1.
            hidden_dim (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent layers in the LSTM.
            output_dim (int): The number of output features to predict. 
                              For predicting a single stock price, this is 1.
        """
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        # batch_first=True causes the input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

        # Define the fully connected linear layer that maps the LSTM's
        # hidden state output to the desired output dimension.
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input sequence for the model. 
                              Shape: (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: The model's prediction. Shape: (batch_size, output_dim)
        """
        # Initialize hidden state and cell state with zeros.
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # We pass the input and the initial hidden & cell states to the LSTM.
        # The LSTM returns the output sequence and the final hidden & cell states.
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))

        # We are interested in the output of the last time step.
        # We use `out[:, -1, :]` to get the last time step's output for each sequence in the batch.
        # Shape: (batch_size, hidden_dim)
        last_time_step_out = out[:, -1, :]

        # Pass the output of the last time step through the linear layer to get the prediction.
        # Shape: (batch_size, output_dim)
        prediction = self.linear(last_time_step_out)
        
        return prediction

# Example of how to instantiate and use the model:
if __name__ == '__main__':
    # Model parameters
    INPUT_DIM = 1
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    OUTPUT_DIM = 1

    # Create a model instance
    model = LSTMForecaster(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
    print("Model Architecture:")
    print(model)

    # Create a dummy input tensor to test the forward pass
    # Shape: (batch_size, sequence_length, input_dim)
    # e.g., a batch of 32 sequences, each 60 time steps long, with 1 feature per step.
    dummy_input = torch.randn(32, 60, INPUT_DIM)

    # Get the model's prediction
    prediction = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Prediction shape: {prediction.shape}")
    # Expected output shape: (32, 1)

