import numpy as np
import numpy.polynomial.polynomial as poly
from sklearn.linear_model import LinearRegression

# Stock price paths from the example
stock_price_paths = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
])

# Option parameters
strike_price = 1.10
risk_free_rate = 0.06
discount_factor = np.exp(-risk_free_rate)  # One-period discount factor

# Number of time steps and paths
n_paths, n_time_steps = stock_price_paths.shape

# Compute intrinsic values (the payoff if exercised immediately)
def put_payoff(stock_prices):
    return np.maximum(strike_price - stock_prices, 0)

# Initialize cash flow matrix
cash_flows = np.zeros((n_paths, n_time_steps))

# Set the cash flows at maturity (t = 3), same as European put option
cash_flows[:, -1] = put_payoff(stock_price_paths[:, -1])

# Backward induction
for t in range(n_time_steps - 2, 0, -1):  # Go from t=2 back to t=1
    in_the_money = stock_price_paths[:, t] < strike_price  # Paths where option is in the money
    
    # If in the money, compute immediate exercise value
    exercise_value = put_payoff(stock_price_paths[:, t])

    # Filter in-the-money paths for regression
    X = stock_price_paths[in_the_money, t]  # Stock prices at time t (in the money)
    Y = cash_flows[in_the_money, t+1] * discount_factor  # Discounted cash flows at t+1
    
    # Perform least squares regression: Y ~ X + X^2
    X_poly = np.vstack([np.ones(X.shape[0]), X, X**2]).T  # Include constant, X, and X^2 terms
    model = LinearRegression().fit(X_poly, Y)
    
    # Estimate continuation values for all paths at time t
    continuation_value = model.predict(np.vstack([np.ones(n_paths), stock_price_paths[:, t], stock_price_paths[:, t]**2]).T)
    
    # Decide whether to exercise or continue
    continuation_value[in_the_money == False] = 0  # No continuation value if not in the money
    cash_flows[:, t] = np.where(exercise_value > continuation_value, exercise_value, cash_flows[:, t+1] * discount_factor)

# At t=0, calculate the option value by discounting expected cash flows
option_value = np.mean(cash_flows[:, 1] * discount_factor)

# Output the option value
print(f"Estimated value of the American put option: {option_value:.4f}")
