import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize

# Define the bounds for the variables (6 variables)
bounds = torch.tensor([
    [0.0, 1.0],  # Variable 1
    [0.0, 5.0],  # Variable 2
    [0.0, 10.0], # Variable 3
    [0.0, 15.0], # Variable 4
    [0.0, 1.0],  # Variable 5
    [0.0, 1.0]   # Variable 6
], dtype=torch.double)

# Constraint: Variable 5 + Variable 6 <= 1
inequality_constraints = [
    (torch.tensor([4, 5], dtype=torch.long), torch.tensor([1.0, 1.0], dtype=torch.double), 1.0)
]

# Define the objective function (toy example)
def toy_objective_function(X):
    return -((X[..., 0] - 0.5)**2 + (X[..., 1] - 3.0)**2 + (X[..., 2] - 7.0)**2 + (X[..., 3] - 10.0)**2 + X[..., 4] + X[..., 5])

def get_my_gp(_x, _y):
    # Define the GP model
    return SingleTaskGP(
        train_X=_x,
        train_Y=_y,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=6, bounds=bounds.T)
    )

# Generate initial training data
train_X = torch.rand(10, 6, dtype=torch.double)  # 10 initial samples in 6 dimensions
train_X[:, 4:] *= 0.5  # Ensure variable 5 and 6 are within constraint bounds
train_Y = toy_objective_function(train_X).unsqueeze(-1)  # Add output dimension

# check if training data is within bounds
print(f'Training data is within bounds: {torch.all(train_X[:, 4:] <= 0.5)}')

gp = get_my_gp(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
mll = fit_gpytorch_mll(mll)
# mll.eval()

# Start the optimization loop
for iteration in range(10):
    # Define the acquisition function
    qEI = qLogExpectedImprovement(model=gp, best_f=train_Y.max())

    # Optimize the acquisition function
    new_Xs, acq_values = optimize_acqf(
        acq_function=qEI,
        bounds=bounds.T,
        q=3, # number of candidates
        inequality_constraints=inequality_constraints,
        num_restarts=10,
        raw_samples=50,
    )
    
    for new_X in new_Xs:
        # Evaluate the new point
        new_Y = toy_objective_function(new_X).unsqueeze(-1)
        print(f'New Value: {new_Y.item()}')

        # Update the training data
        train_X = torch.cat([train_X, new_X.unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)], dim=0)

    # Refit the GP model
    gp = get_my_gp(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll = fit_gpytorch_mll(mll)

# Print final results
print("Optimized Input:", train_X[train_Y.argmax()])
print("Optimized Objective Value:", train_Y.max().item())
