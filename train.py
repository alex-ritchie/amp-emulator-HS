import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from hyperamp import ODEFunc_base, ODEFunc_correction, NeuralODE, Hypersolver

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(X, Y, config):
    # first order Neural ODE of Hypersolver
    ode_func_base = ODEFunc_base(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'])
    neural_ode_base = NeuralODE(ode_func_base)
    # Second order Neural ODE of Hypersolver
    ode_func_correction = ODEFunc_correction(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'])
    neural_ode_correction = NeuralODE(ode_func_correction)
    # Initialize Hypersolver
    hs = Hypersolver(neural_ode_base, neural_ode_correction)

    # Training Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(hs.parameters(), lr=config['model']['learning_rate'])

    # Training loop
    num_epochs = config['model']['num_epochs']
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        t = torch.linspace(0, 1, X.size(0))
        pred_y = hs(X, t)
        loss = criterion(pred_y, Y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print("Training complete.")

if __name__ == '__main__':
    # Load configuration
    config = load_config('config.yaml')

    # Example dataset (replace with your actual dataset)
    # X: input features, Y: target outputs
    X = torch.randn(100, config['model']['input_dim'])
    Y = torch.randn(100, config['model']['output_dim'])

    main(X, Y, config)