import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_synthetic_data():
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_vals = X_grid * np.sin(Y_grid)

    X_data = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z_data = Z_vals.ravel()

    return X_data, Z_data


def scale_features(X_train_data, X_test_data):
    scaler = StandardScaler()
    X_train_scaled = torch.FloatTensor(scaler.fit_transform(X_train_data))
    X_test_scaled = torch.FloatTensor(scaler.transform(X_test_data))
    return X_train_scaled, X_test_scaled


def preprocess_data():
    X_data, Z_data = create_synthetic_data()
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, Z_data, test_size=0.1)

    X_train_scaled, X_test_scaled = scale_features(X_train_data, X_test_data)
    y_train_scaled = torch.FloatTensor(y_train_data)
    y_test_scaled = torch.FloatTensor(y_test_data)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled


class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes, net_type='standard'):
        super(CustomNetwork, self).__init__()
        self.net_type = net_type
        input_dim = 2
        prev_dim = input_dim
        layers_list = []
        
        if net_type == 'standard':
            for size in layer_sizes:
                layers_list.append(nn.Linear(prev_dim, size))
                layers_list.append(nn.Tanh())
                prev_dim = size
            layers_list.append(nn.Linear(prev_dim, 1))
            self.model = nn.Sequential(*layers_list)
        elif net_type == 'cascade':
            self.hidden_layers = nn.ModuleList()
            for size in layer_sizes:
                self.hidden_layers.append(nn.Linear(prev_dim + input_dim, size))
                prev_dim = size
            self.output_layer = nn.Linear(prev_dim + input_dim, 1)
        elif net_type == 'elman':
            self.hidden_layer_sizes = layer_sizes  
            self.input_to_hidden = nn.ModuleList([nn.Linear(input_dim + size, size) for size in layer_sizes])
            self.hidden_to_output = nn.Linear(self.hidden_layer_sizes[-1], 1)
            self.tanh_activation = nn.Tanh()

    def forward(self, input_data, hidden_state=None):
        if self.net_type == 'standard':
            return self.model(input_data)
        elif self.net_type == 'cascade':
            output = input_data
            for layer in self.hidden_layers:
                output = torch.cat([input_data, output], dim=1)
                output = torch.tanh(layer(output))
            output = torch.cat([input_data, output], dim=1)
            return self.output_layer(output)
        elif self.net_type == 'elman':
            if hidden_state is None:
                hidden_state = self.initialize_hidden(input_data.size(0))
                
            combined_input = torch.cat((input_data, hidden_state[0]), 1)
            hidden_next = []
            for i, input_to_hidden_layer in enumerate(self.input_to_hidden):
                hidden_step = self.tanh_activation(input_to_hidden_layer(combined_input))
                combined_input = torch.cat((input_data, hidden_step), 1)
                hidden_next.append(hidden_step)
            output = self.hidden_to_output(hidden_next[-1])
            return output, hidden_next

    def initialize_hidden(self, batch_size):
        return [torch.zeros(batch_size, size) for size in self.hidden_layer_sizes]


def train_custom_model(model, epochs, learning_rate, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    model.apply(initialize_weights)

    for epoch in range(epochs):
        model.train()
        if isinstance(model, CustomNetwork) and model.net_type == 'elman':
            hidden_state = model.initialize_hidden(X_train_scaled.size(0))
            output, hidden_state = model(X_train_scaled, hidden_state)
        else:
            output = model(X_train_scaled)

        loss = loss_function(output, y_train_scaled.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())
        if epoch % 1000 == 0:
            print(f'At Epoch {epoch}, the training loss is: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        if isinstance(model, CustomNetwork) and model.net_type == 'elman':
            hidden_state = model.initialize_hidden(X_test_scaled.size(0))
            test_predictions, hidden_state = model(X_test_scaled, hidden_state)
        else:
            test_predictions = model(X_test_scaled)
        mre = torch.mean(torch.abs((y_test_scaled - test_predictions.squeeze()) / y_test_scaled)).item()

    return test_predictions, mre, loss_history


def display_results(test_predictions, y_test_scaled, loss_history, net_type, config_label):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_scaled, test_predictions.numpy().squeeze(), color='red')
    plt.plot([-3, 3], [-3, 3], color='blue', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actuals ({net_type} {config_label})')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(loss_history, color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({net_type} {config_label})')
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.show()


def run_training():
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = preprocess_data()

    network_configurations = {
        '1L 10U Standard': {'net_type': 'standard', 'layer_sizes': [10]},
        '1L 20U Standard': {'net_type': 'standard', 'layer_sizes': [20]},
        '1L 20U Cascade': {'net_type': 'cascade', 'layer_sizes': [20]},
        '2L 10U Cascade': {'net_type': 'cascade', 'layer_sizes': [10, 10]},
        '1L 15U Elman': {'net_type': 'elman', 'layer_sizes': [15]},
        '3L 5U Elman': {'net_type': 'elman', 'layer_sizes': [5, 5, 5]},
    }

    for config_label, config in network_configurations.items():
        print(f"\nTraining with Config: {config_label}")
        model = CustomNetwork(**config)
        test_predictions, mre, loss_history = train_custom_model(model, epochs=6000, learning_rate=0.01, 
                                                                X_train_scaled=X_train_scaled, y_train_scaled=y_train_scaled,
                                                                X_test_scaled=X_test_scaled, y_test_scaled=y_test_scaled)
        print(f'Mean Relative Error: {mre:.4f}')
        display_results(test_predictions, y_test_scaled, loss_history, model.net_type, config_label)


run_training()
