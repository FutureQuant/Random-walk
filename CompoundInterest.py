import numpy as np
#set random seed
np.random.seed(42)
#1M random price changes (daily, Mean = 0, std = 0.01)
n = 1_000_000
daily_returns = np.random.normal(loc=0, scale=0.01, size=n)

#create price series
prices = 100 * np.cumprod(1 + daily_returns)

#20 day moving average
window = 20
moving_avg = np.convolve(prices, np.ones(window)/window, mode='valid')

#sample outputs
print(f"First 10 prices: {prices[:10]}")
print(f"First 10 moving averages: {moving_avg[:10]}")
print(f"Total prices: {len(prices)}, Total MAs: {len(moving_avg)}")

np.savetxt("prices.csv", prices, delimiter=",")
np.savetxt("moving_avg.csv", moving_avg, delimiter=",")
