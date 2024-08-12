import numpy as np
import pandas as pd

from search import BayesianSymbolicRegressor

def generate_example_data():
    # EQUATION | y = x0**2 - x1**2 + x1**2 - 1
    x0 = np.random.uniform(low=-100, high=100, size=500)
    x1 = np.random.uniform(low=-100, high=100, size=500)
    y = x0**2 - x1**2 + x1**2 - 1

    X = pd.DataFrame({
        "x0": x0,
        "x1": x1
    })
    y = pd.Series(y)

    return X, y

def generate_hooks_law_data():
    k = np.random.uniform(low=1, high=100, size=50)
    original_length = np.random.uniform(low=1, high=100, size=50)
    current_length = original_length + np.random.uniform(low=1, high=25, size=50)

    f = k * (current_length - original_length)


    X = pd.DataFrame({
        "k": k,
        "original_lenght": original_length,
        "current_length": current_length
    })
    y = pd.Series(f)

    return X, y

if __name__ == "__main__":
    #X, y = generate_example_data()
    X, y = generate_hooks_law_data()
    
    """
    df = pd.read_csv("../data/london_merged.csv")
    num_lags = 15
    for lag_number in range(1, 15 + 1):
        df[f"cnt_lag_{lag_number}"] = df["cnt"].shift(lag_number)
    df = df.dropna(axis=0)
    X = df.drop(["cnt", "timestamp","t1", "t2", "hum", "wind_speed", "weather_code", "is_holiday", "is_weekend", "season"], axis=1)
    y = df["cnt"]
    """

    model = BayesianSymbolicRegressor(
        max_tree_depth=5,
        timeout=600,
        n_iterations=5000,
        use_constants=False,
        metric="mae",
        stopping_criteria=1e-3
    )
    model.fit(X, y)
    predictions = model.predict(X)

    print("PREDICTIONS:")
    print(predictions)
    print()
    print("GROUND TRUTH:")
    print(y)
    print()
    print("BEST EQUATION:")
    print(model.best_equation)

    print("MAE:")
    print(np.mean(np.abs(predictions - y.values)))
