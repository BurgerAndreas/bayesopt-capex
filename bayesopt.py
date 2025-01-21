import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import scipy.optimize as sciopt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

from data import parameter_bounds, parameter_granularity, stock_concentrations, variable_order

"""

Future work:
- handle different number of compounds, 
by setting unused compounts to concentration=0 in observations, 
and setting concentration to 0 for predicted concentrations

"""


class BayesianOptimizer:
    def __init__(self, X_train=None, y_train=None, bounds=None, granularity=None, enforce_constraint=False):
        """
        Initialize the Bayesian Optimizer with optional training data.
        
        Parameters:
        bounds (list of tuples): List of (min, max) tuples for each variable
        granularity (list of floats): List of granularity for each variable
        X_train (numpy.ndarray): Initial training data features. shape: (n_samples, n_vars)
        y_train (numpy.ndarray): Initial training data targets. shape: (n_samples, 1)
        enforce_constraint (bool): Whether to enforce the constraint that variables 4,5,6 sum to 1 # TODO: change
        """
        self.bounds = bounds
        self.granularity = granularity
        self.enforce_constraint = enforce_constraint
        self.n_vars = len(bounds)
        
        # Initialize scalers for inputs and outputs
        # each column is scaled individually
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Initialize storage for observations
        self.X_observed = []
        self.y_observed = []
        
        # If training data is provided, initialize with it
        if X_train is not None and y_train is not None:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            if len(X_train.shape) == 1:
                X_train = X_train.reshape(1, -1)
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
                
            # Fit scalers with training data
            self.X_scaler.fit(X_train)
            self.y_scaler.fit(y_train)
            
            # Add training data to observations
            for i in range(len(X_train)):
                self.X_observed.append(X_train[i])
                self.y_observed.append(y_train[i].ravel()[0])
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0] * self.n_vars)
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42
        )
    
    def _normalize_X(self, X):
        """Normalize input variables to [0, 1] range"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.X_scaler.transform(X)
    
    def _denormalize_X(self, X_norm):
        """Convert normalized variables back to original scale"""
        return self.X_scaler.inverse_transform(X_norm)
    
    def _normalize_y(self, y):
        """Normalize output variable"""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if not hasattr(self, 'y_min') or not hasattr(self, 'y_max'):
            self.y_scaler.fit(y)
        return self.y_scaler.transform(y)
    
    def _denormalize_y(self, y_norm):
        """Convert normalized output back to original scale"""
        return self.y_scaler.inverse_transform(y_norm)
    
    def _acquisition_function(self, X, xi=0.01):
        """
        Compute the Expected Improvement acquisition function.
        
        Parameters:
        X: Points to evaluate EI at
        xi: Exploration-exploitation trade-off parameter
        """
        X_norm = self._normalize_X(X)
        
        if len(X_norm.shape) == 1:
            X_norm = X_norm.reshape(1, -1)
            
        mu, sigma = self.gpr.predict(X_norm, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        mu_sample = self._normalize_y(np.array([max(self.y_observed)]).reshape(-1, 1))
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample - xi
            Z = imp / sigma
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei.ravel()
    
    def _enforce_constraint(self, point):
        """We need to incorporate the following constraints:
        - C_des_i <= C_stock_i for all i
        - sum_i C_des_i / C_stock_i <= 1
        where C_des_i is the desired concentration of the i-th compound
        and C_stock_i is the stock concentration of the i-th compound
        """
        point = point.copy()
        
        # concentrations are variables are 4:
        predicted_concentrations = point[4:]
        
        # clip variables to the bounds
        # shouldn't be necessary since we're using the bounds in the optimizer
        # for i in range(len(point)):
        #     if i < 4:
        #         point[i] = np.clip(point[i], self.bounds[i][0], self.bounds[i][1])
        
        # sum of relative concentrations must be less than or equal to 1
        # relative concentration is defined as C_des_i / C_stock_i
        relative_concentrations = predicted_concentrations / stock_concentrations
        if np.sum(relative_concentrations) > 1:
            print("Sum of relative concentrations exceeds 1, clipping to 1")
            predicted_concentrations /= np.sum(relative_concentrations)
        
        # TODO: make list of bucket boundaries instead?
        # bucket the variables to the correct granularity
        for i in range(self.n_vars):
            if self.granularity[i] != -1:
                point[i] = round(point[i] / self.granularity[i]) * self.granularity[i]
        
        return point
    
    def suggest_next_point(self):
        """Suggest the next point to evaluate"""
        if len(self.X_observed) < 2:  # Need at least 2 points to fit GP
            # # Random sampling if not enough points
            # next_point = np.array([
            #     np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            #     for i in range(self.n_vars)
            # ])
            raise ValueError("Not enough points to fit GP")
        
        # Use Bayesian optimization to suggest next point
        def objective(X):
            return -self._acquisition_function(X.reshape(1, -1))
        
        best_x = None
        best_acquisition = -np.inf
        
        # Random restart optimization to avoid local minima
        n_restarts = 25
        starting_points = [
            np.random.uniform(self.bounds[i][0], self.bounds[i][1], self.n_vars) 
            for i in range(self.n_vars)
        ]
        for starting_point in starting_points:
            res = sciopt.minimize(
                objective,
                starting_point,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if -res.fun >= best_acquisition:
                best_acquisition = -res.fun
                best_x = res.x
        
        next_point = best_x
        
        # TODO: do we need to denormalize here?
        # next_point = self._denormalize_X(next_point)
        
        if self.enforce_constraint:
            next_point = self._enforce_constraint(next_point)
            
        return next_point
    
    def observe(self, X, y):
        """
        Add an observation to the optimizer.
        
        Parameters:
        X: Input variables
        y: Observed output
        """
        X = np.asarray(X)
        y = np.asarray(y)
            
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        self.X_observed.append(X.ravel())
        self.y_observed.append(y.ravel()[0])
        
        # Update GP model if we have enough points
        if len(self.X_observed) >= 2:
            X_train_norm = self._normalize_X(np.vstack(self.X_observed))
            y_train_norm = self._normalize_y(np.array(self.y_observed).reshape(-1, 1))
            self.gpr.fit(X_train_norm, y_train_norm.ravel())
    
    def get_best_observation(self):
        """Return the best observed point and its value"""
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

# Example usage
if __name__ == "__main__":
    """Test if the BayesOpt model converges to a minimum.
    """
    
    # our bounds do not work for the fake objective function
    # bounds = [parameter_bounds[k] for k in variable_order]
    bounds = [
        (-10, 10),  # Bounds for variable 1
        (-5, 5),    # Bounds for variable 2
        (0, 15),    # Bounds for variable 3
        (0, 1),     # Bounds for variable 4 
        (0, 1),     # Bounds for variable 5 
        (0, 1)      # Bounds for variable 6 
    ]
    
    # TODO: not used at the moment
    granularity = [parameter_granularity[k] for k in variable_order]
    
    # Generate some example training data
    np.random.seed(42)
    X_train = np.array([
        [np.random.uniform(bound[0], bound[1]) for bound in bounds]
        for _ in range(20)
    ])
    
    # Example objective function 
    # this would be the actual experiment
    def objective_function(x):
        # nice convex function
        return -(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2)
    
    # Generate training targets
    y_train = np.array([objective_function(x) for x in X_train])
    
    # Initialize optimizer with training data
    optimizer = BayesianOptimizer(X_train, y_train, bounds=bounds, granularity=granularity)
    
    # Run optimization
    # i.e. run the suggested experiments
    n_iterations = 30
    for i in range(n_iterations):
        # Get next point to evaluate
        next_point = optimizer.suggest_next_point()
        
        # Evaluate objective function
        value = objective_function(next_point)
        
        # Add observation
        optimizer.observe(next_point, value)
        
        # Print progress
        if (i + 1) % 10 == 0:
            best_x, best_y = optimizer.get_best_observation()
            print(f"Iteration {i+1}: Best value = {best_y:.4f}")
            print(f"Best parameters: {best_x}")