import pandas as pd
import yfinance as yf
import statsmodels.api as sm


def main(data):
    tickers = []
    for column in data.columns:
        if column not in ['Mkt-RF', 'SMB', 'HML', 'RF']:
            tickers.append(column)
    
    stocks_excess = data[tickers].sub(data['RF'], axis=0)
    X = data[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    results_list = []
    for ticker in tickers:
        Y = stocks_excess[ticker]
        regression_data = pd.concat([Y, X], axis=1).dropna()
        Y_clean = regression_data[ticker]
        X_clean = regression_data[['const', 'Mkt-RF', 'SMB', 'HML']]
        model = sm.OLS(Y_clean, X_clean)
        results = model.fit()
        results_list.append({
            'ticker': ticker,
            'alpha': results.params['const'],
            'alpha_se': results.bse['const'],
            'alpha_t': results.tvalues['const'],
            'beta': results.params['Mkt-RF'],
            'beta_se': results.bse['Mkt-RF'],
            'smb_loading': results.params['SMB'],
            'smb_se': results.bse['SMB'],
            'hml_loading': results.params['HML'],
            'hml_se': results.bse['HML'],
            'nobs': results.nobs
        })
    return pd.DataFrame(results_list)




if __name__ == '__main__':
    start_date = "2020-12-31"
    end_date = "2026-01-01"

    stocks = pd.read_csv('data/selected_stocks.csv')

    tickers = stocks["Symbol"].tolist()

    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        )
    #有些股票没数据。
    close_prices = prices["Close"]
    missing_days = close_prices.isna().sum()
    keep_mask = missing_days <= 100
    close_prices = close_prices.loc[:, keep_mask]
    returns = close_prices.pct_change()
    #print(returns.isna().sum().to_list()) 
    #print(returns[returns.isna().any(axis=1)]) 
    returns = returns.dropna(axis=0, how = 'all')
    #print(returns.head(20))

    factors = pd.read_csv('data/F-F_Research_Data_Factors_daily.csv')
    factors['Date'] = pd.to_datetime(factors['Date'],format='%Y%m%d')
    factors = factors.set_index('Date')
    factors = factors / 100
    #print(factors.head())

    data = returns.join(factors, how='inner')

    print(data.head())
    print(main(data).head(20))
    data.to_csv('data/all_time_data.csv', index=True, index_label="Date")