import numpy as np

def simpsons_rule(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using Simpson's 1/3 rule.
    
    Parameters:
        f : function
            The function to integrate.
        a : float
            The lower limit of integration.
        b : float
            The upper limit of integration.
        n : int
            The number of subintervals (must be even).
    
    Returns:
        float
            Approximate integral value.
    """
    if n % 2 != 1:
        raise ValueError("n must be even for Simpson's rule")
    
    h = (b - a) / n  # Step size
    x = np.linspace(a, b, n + 1)
    
    integral = y[0] + y[-1]  # First and last terms
    integral += 4 * sum(f[1:n:2])  # Odd index terms (coefficient 4)
    integral += 2 * sum(f[2:n-1:2])  # Even index terms (coefficient 2)
    
    return (h / 3) * integral

# Example usage
if __name__ == "__main__":
    from math import sin, pi
    
    # Define function to integrate
    def func(x):
        return sin(6*x)**4
    
    # Integration limits
    a, b = 0, pi / 2
    n = 10  # Number of subintervals (must be even)
    
    result = simpsons_rule(func, a, b, n)
    print(f"Approximate integral: {result}")