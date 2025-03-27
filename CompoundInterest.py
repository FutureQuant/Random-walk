import numpy as np

def simulate_prices(n=1_000_000, start_price=100, mean_return=0, std_return=0.01, window=20, seed=42):
    """
    Efficiently simulate price series with moving average calculation.
    
    Parameters:
    - n: Number of daily returns to simulate
    - start_price: Initial price
    - mean_return: Mean of daily returns
    - std_return: Standard deviation of daily returns
    - window: Moving average window size
    - seed: Random seed for reproducibility
    
    Returns:
    - prices: Simulated price series
    - moving_avg: Moving average of prices
    """
    # Use NumPy's random generator for better performance
    rng = np.random.default_rng(seed)
    
    # Generate daily returns more efficiently
    daily_returns = rng.normal(loc=mean_return, scale=std_return, size=n)
    
    # Compute cumulative product in-place for memory efficiency
    prices = start_price * np.cumprod(1 + daily_returns)
    
    # Use more efficient moving average calculation
    moving_avg = np.convolve(prices, np.ones(window)/window, mode='valid')
    
    return prices, moving_avg

def save_to_csv(data, filename):
    """
    Save NumPy array to CSV with error handling.
    
    Parameters:
    - data: NumPy array to save
    - filename: Output CSV filename
    """
    try:
        np.savetxt(filename, data, delimiter=",")
        print(f"Successfully saved {filename}")
    except IOError as e:
        print(f"Error saving {filename}: {e}")

def main():
    # Simulate prices
    prices, moving_avg = simulate_prices()
    
    # Print sample outputs with more informative formatting
    print(f"First 10 prices: {prices[:10]}")
    print(f"First 10 moving averages: {moving_avg[:10]}")
    print(f"Total prices: {len(prices)}, Total Moving Averages: {len(moving_avg)}")
    
    # Optional: Save to files
    save_to_csv(prices, "prices.csv")
    save_to_csv(moving_avg, "moving_avg.csv")

if __name__ == "__main__":
    main()
