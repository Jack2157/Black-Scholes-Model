import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T))- r * K * np.exp(-r * T) * norm.cdf(d2))
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)

    return price, delta, gamma, vega, theta, rho


price, delta, gamma, vega, theta, rho = black_scholes(100, 100, 1, 0.05, 0.2)
print(f"Price: {price:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega:  {vega:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Rho:   {rho:.4f}")


spots = np.linspace(60, 140, 50)
vols  = np.linspace(0.1, 0.6, 50)
S_grid, V_grid = np.meshgrid(spots, vols)

prices = black_scholes(S_grid, 100, 1, 0.05, V_grid)[0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(S_grid, V_grid, prices, cmap='viridis')
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Volatility')
ax.set_zlabel('Call Price')
ax.set_title('Call Price Surface')
plt.show()
