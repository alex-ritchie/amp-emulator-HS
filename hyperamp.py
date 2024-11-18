import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# TODO: specify ode func architecture for base and correction
# Define the neural ODE function
class ODEFunc_base(nn.Module):
    def __init__(self):
        super(ODEFunc_base, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        return self.net(y)
    
# Define the neural ODE function
class ODEFunc_correction(nn.Module):
    def __init__(self):
        super(ODEFunc_correction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        return self.net(y)

# TODO: implement helper function for training

# Define the neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func

    def forward(self, X, t):
        return odeint(self.ode_func, X, t)

# Hypersolver (Boosting approach)
class Hypersolver(nn.Module):
    def __init__(self, base_solver, correction_solver):
        super(Hypersolver, self).__init__()
        self.base_solver = base_solver
        self.correction_solver = correction_solver

    def forward(self, X, t):
        base_solution = self.base_solver(X, t)
        correction = self.correction_solver(X, t)
        return base_solution + correction
