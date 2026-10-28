import pandas as pd

df = pd.read_csv('data/universe_raw.csv')
badkeywords = 'Preferred|Preference|Notes|Warrant|Corporate Units|Equity Units'
exclude_mask = df['Name'].str.contains(badkeywords, case = False)
df = df[~exclude_mask]
handselectbadtickers = ['CCZ', 'GOOG', 'LILAK', 'GLIBK', 'EPD', 'ET', 'CQP', 'MPLX', 'PAA', 'SUN']
exclude_mask = df['Symbol'].isin(handselectbadtickers)
df = df[~exclude_mask]

sectors = []

for seci in df['Sector'].dropna():
    if seci not in sectors:
        sectors.append(seci)

def sector_sorter(sector):
    sector_mask = df['Sector'] == sector
    sector_stocks = df[sector_mask]
    sector_stocks_sorted = sector_stocks.sort_values('Market Cap', ascending = False)
    return sector_stocks_sorted.head(30)

candidates = []
for seci in sectors:
    sector_candidates = sector_sorter(seci)
    for i in sector_candidates.index:
        candidates.append({
            'Sector' : sector_candidates['Sector'][i],
            'Symbol' : sector_candidates['Symbol'][i],
            'Name' : sector_candidates['Name'][i],
            'Market Cap' : sector_candidates['Market Cap'][i]
        })

selected_stocks = pd.DataFrame(candidates)
#print(selected_stocks)
#print(selected_stocks.to_string(index=False))

selected_stocks.to_csv('data/selected_stocks.csv', index=False)
