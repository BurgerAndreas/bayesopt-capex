import torch

inequality_constraints = [
    (0,1,2), # indices of the variables
    -1 * torch.tensor([1.0/0.4, 1.0/0.4, 1.0]), # coefficients of the variables
    -1.0 # right-hand side
]

def check_constrains(values):
    indices, coefficients, rhs = inequality_constraints
    _sum = torch.sum(coefficients * values[torch.tensor(indices)])
    formula = ''.join([f'{coefficients[i]}*{values[indices[i]]:.3f} + ' for i in range(len(coefficients))]) + f'= {_sum} >= {rhs} ({_sum >= rhs})'
    print(formula)

# bayesopt wants the form
# sum (coefficients[i] * variables[indices[i]]) >= right-hand side
# we want the constraint a+b+c <= 1
# that is why we multiply the coefficients and the rhs by -1

# test 1: should fail
values = [0.2, 0.21, 0.1]
check_constrains(torch.tensor(values))

# test 2: should pass
values = [0.1, 0.1, 0.1]
check_constrains(torch.tensor(values))
