import pandas as pd

def make_pfl_1(resdf):
    ranked = resdf.sort_values('alpha_t', ascending = False).reset_index()
    long_stocks = ranked.head(20)
    short_stocks = ranked.tail(20)
    beta_long = long_stocks["beta"].mean()
    beta_short = short_stocks["beta"].mean()
    long_w = beta_short / (beta_long + beta_short)
    short_w = -beta_long / (beta_long + beta_short)
    long_stocks["weight"] = long_w / 20
    short_stocks["weight"] = short_w / 20
    portfolio = pd.concat([long_stocks, short_stocks],ignore_index=True,)
    return portfolio[["ticker", "weight"]].copy()

if __name__ == "__main__":
    from main import main
    data = pd.read_csv('data/all_time_data.csv')
    portfolio = make_pfl_1(main(data))
    print(portfolio)