import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from main import main
from strategy_1 import make_pfl_1

data = pd.read_csv('data/all_time_data.csv')
data["Date"] = pd.to_datetime(data["Date"])
data = data.set_index("Date").sort_index()

def backtest_1(data,rebatime,lookback = 252):
    return_list = []
    return_daily_list = []
    for end in range(lookback, len(data), rebatime):
        start = end - lookback
        window_data = data.iloc[start:end]
        window_results = main(window_data)
        window_pfl = make_pfl_1(window_results)
        window_pfl = window_pfl.set_index("ticker")
        holding_data = data.iloc[end:end + rebatime]

        growth = pd.DataFrame(index=holding_data.index)
        holding_value = pd.Series(1.0, index=holding_data.index)

        for ticker in window_pfl.index:
            growth[ticker] = (1 + holding_data[ticker]).cumprod()
            holding_value += window_pfl['weight'][ticker] * (growth[ticker] - 1)

        return_list.append(float((holding_value.iloc[-1] - 1) * 100))
        daily_return = holding_value.pct_change()
        daily_return.iloc[0] = holding_value.iloc[0] - 1 #manually update the first day return. or it would be NaN
        for day in holding_data.index:
            return_daily_list.append({
                'date' : day,
                'return' : float(daily_return.loc[day]),
            })
    
    return(return_list, return_daily_list)

return_list, return_daily_list, = backtest_1(data, 5)
results_daily = pd.DataFrame(return_daily_list)
results_daily = results_daily.set_index("date")

results_daily["equity"] = (1 + results_daily["return"]).cumprod()

daily_volatility = results_daily["return"].std()
annual_volatility = daily_volatility * np.sqrt(252)
cumulative_return = results_daily["equity"].iloc[-1] - 1

print(f"Cumulative Return: {cumulative_return:.2%}")
print(f"Annualized Volatility: {annual_volatility:.2%}")

results_daily["max_so_far"] = results_daily["equity"].cummax().clip(lower=1)
results_daily["drawdown"] = (
    1 - results_daily["equity"] / results_daily["max_so_far"]
)
print(f"Maximum Drawdown: {results_daily['drawdown'].max():.2%}")


results_daily["excess_return"] = results_daily["return"] - data["RF"]

annual_sharpe = (
    results_daily["excess_return"].mean()
    / results_daily["excess_return"].std()
    * np.sqrt(252)
)
print(f"Annualized Sharpe Ratio: {annual_sharpe:.2f}")


final_regression_data = pd.DataFrame({
    "portfolio": results_daily["excess_return"],
    "market": data["Mkt-RF"],
}).dropna()

X = sm.add_constant(final_regression_data["market"])
Y = final_regression_data["portfolio"]

model = sm.OLS(Y, X)
final_results = model.fit()

print(f"Realized Beta: {final_results.params['market']:.4f}")
print(f"Beta Standard Error: {final_results.bse['market']:.4f}")


plt.plot(results_daily.index, results_daily["equity"])
plt.title("FF3 Market-Neutral Portfolio")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.show()