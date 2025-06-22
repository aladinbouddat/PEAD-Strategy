import sys
#print("Python path used:", sys.executable)
import matplotlib
import matplotlib.dates as mdates
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('TkAgg')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Backtester:
    def __init__(self, mydata, myprices, index_prices, t_bill, sue_thresholds, days_l, days_s, initial_cash=100000):
        self.extreme_ticker_list = []
        self.data = mydata
        self.date = None
        self.prices = myprices
        self.spy = index_prices
        self.treasury_3m = t_bill
        self.sue_thresholds = sue_thresholds
        self.init_cash = initial_cash
        self.cash = initial_cash
        self.equity = 0
        self.equity_bm = 0
        self.total_value_bm = initial_cash
        self.cash_bm = 0
        self.spy_shares = self.init_cash / spy.loc[spy['Date'] == initial_date.strftime("%Y-%m-%d")]['Close_Price'].iloc[0]
        self.positions = positions
        self.portfolio = portfolio
        self.margin_account = 0
        self.total_pnl = 0
        self.total_pnl_bm = 0
        self.daily_return = 0
        self.daily_return_bm = 0
        self.unrealized_pnl = 0
        self.pnl_from_short = 0
        self.pnl_from_long = 0
        self.close_stack = close_stack
        self.days_short = days_s
        self.days_long = days_l
        self.transaction_cost = 0
        self.transaction_cost_bm = 0
        self.counter = 0
        self.trade_volume = 0
        self.traded_shares = 0
        self.trade_volume_bm = 0
        self.traded_shares_bm = 0

    def execute_order(self, signal, ticker, price, order_size, current_position):
        order_value = price * order_size
        transaction_cost = self.get_transaction_cost(order_size, order_value, signal)

        if signal == 'Open Long':
            self.cash -= order_value
            self.positions[ticker]['Shares'] += order_size
            self.positions[ticker]['Entry'] = price
            self.cash -= transaction_cost
            self.transaction_cost += transaction_cost
            # print(f"{self.date.date()}, Open Long, {ticker}\n   Cash: -{round(order_size * price)}, Positions: +{round(order_size * price)}, Margin: No Change")

        elif signal == 'Open Short':
            self.cash -= margin * order_value
            self.margin_account += margin * order_value
            self.positions[ticker]['Shares'] -= order_size
            self.positions[ticker]['Entry'] = price
            self.cash -= transaction_cost
            self.transaction_cost += transaction_cost
            # print(f"{self.date.date()}, Open Short, {ticker} Order-Size: {round(order_size)}, Price: {round(price, 2)}\n   Cash: -{round(margin * order_size * price)}, Positions: No Big Change, Margin: +{round(price * order_size * (1 + self.margin))}")

        elif signal == 'Close Long':
            self.positions[ticker]['Exit'] = price
            self.cash += price * current_position
            self.positions[ticker]['Shares'] -= current_position
            self.total_pnl += current_position * (self.positions[ticker]['Exit'] - self.positions[ticker]['Entry'])
            self.pnl_from_long += current_position * (self.positions[ticker]['Exit'] - self.positions[ticker]['Entry'])
            self.cash -= transaction_cost
            self.transaction_cost += transaction_cost
            # print(f"{self.date.date()}, Close Long, {ticker}\n   Cash: +{round(price * current_position)}, Positions: -{round(price * current_position)}, Margin: No Change, Entry: {round(self.positions[ticker]['Entry'], 2)}, Exit: {round(exit_price, 2)}, Profit: {round(current_position * (exit_price - entry_price))}")

        elif signal == 'Close Short':
            self.positions[ticker]['Exit'] = price
            profit_loss = (self.positions[ticker]['Entry'] - self.positions[ticker]['Exit']) * abs(current_position)
            self.margin_account -= (margin * self.positions[ticker]['Entry'] * abs(current_position))
            self.cash += margin * self.positions[ticker]['Entry'] * abs(current_position) + profit_loss
            self.positions[ticker]['Shares'] += abs(current_position)
            self.total_pnl += profit_loss
            self.pnl_from_short += profit_loss
            self.cash -= transaction_cost
            self.transaction_cost += transaction_cost

            # print(f"{self.date.date()}, Close Short, {ticker}\n   Cash: +{round(margin * entry_price * abs(current_position) + profit_loss)}, Positions: No Big Change, Margin: -{round(entry_price * abs(current_position) * (1 + self.margin))}, Entry: {round(entry_price, 2)}, Exit: {round(exit_price, 2)}, Current Position: {round(current_position, 2)}, Profit: {round((entry_price - exit_price) * abs(current_position))}")

        self.trade_volume += order_value
        self.traded_shares += order_size
        # if self.date.day == 28:
            # print(f"{self.date.date()} {signal}: {round(order_size, 1)} shares of {ticker} at ${round(price, 1)}, total: ${round(order_value)}, Total Profit Portfolio: ${round(self.total_pnl)}, vs. Benchmark: ${round(self.total_pnl_bm)}")
        # print(f"{self.date.date()} Cash: {round(self.cash)}, Positions: {round(self.equity)}, Margin: {round(self.margin_account)}, Total: {round(self.cash + self.equity)}, Unrealized Profits: {round(self.unrealized_pnl, 2)}, PNL from Long: {round(self.pnl_from_long)}, PNL from Short: {round(self.pnl_from_short)}, Total PNL: {round(self.total_pnl)}")
        self.counter += 1

    def get_transaction_cost(self, order_size, order_value, signal):
        spread = spread_median * order_value
        ib_commission = max(ib_comission * order_size, min_ib_commission)
        third_party_fee = third_party_fees * order_size
        pass_through_fee = pass_through_factor * ib_commission
        total_cost = spread + ib_commission + third_party_fee + pass_through_fee

        if signal == "Open Long":
            return total_cost
        elif signal in ["Open Short", "Close Long"]:
            regulatory_selling_fees = finra_selling_fee + sec_selling_fee * order_value
            return total_cost + regulatory_selling_fees
        elif signal == "Close Short":
            borrow_cost = stock_borrowing_cost * order_value * (self.days_short / 360)
            return total_cost + borrow_cost

    def add_close_date(self, signal, ticker, price):
        if signal == "Close Long":
            days = self.days_long
        else:
            days = self.days_short
        if self.date > self.data.iloc[-1, 0] - timedelta(days):
            close_date = self.data.iloc[-1, 0]
        else:
            close_date = adjust_to_weekday(self.date + timedelta(days=days))
        shares = self.positions[ticker]['Shares']
        new_row = pd.DataFrame(
            [[close_date, ticker, price, shares, signal]], columns=['Date', 'Ticker', 'Price', 'Shares', 'Signal'])
        new_row['Ticker'] = new_row['Ticker'].astype("string")
        new_row['Signal'] = new_row['Signal'].astype("string")
        self.close_stack = pd.concat([self.close_stack, new_row]).reset_index(drop=True)

    def calculate_equity_value(self):
        equity = 0
        self.unrealized_pnl = 0
        prices_for_date = self.prices[self.prices['Date'] == self.date]
        # Convert self.positions dictionary to DataFrame
        positions_df = pd.DataFrame.from_dict(self.positions, orient='index').reset_index()
        positions_df = positions_df.rename(columns={'index': 'Ticker'})
        long_positions_df = positions_df.loc[positions_df['Shares'] > 0]
        if not long_positions_df.empty:
            merged_long = pd.merge(long_positions_df[['Ticker', 'Shares', 'Entry']],
                                   prices_for_date[['Ticker', 'Close_Price']], on='Ticker', how='inner')
            merged_long = merged_long.dropna()
            merged_long['Product'] = merged_long['Close_Price'] * merged_long['Shares']
            merged_long['Unrealized PNL'] = (merged_long['Close_Price'] - merged_long['Entry']) * merged_long['Shares']
            self.unrealized_pnl += merged_long['Unrealized PNL'].sum()
            self.unrealized_pnl = 0
            equity += merged_long['Product'].sum()
            #merged_long_extreme = merged_long[(merged_long['Close_Price'] / merged_long['Entry'] - 1) * 100  > 500]
            #if not merged_long_extreme.empty:
                #self.extreme_ticker_list.extend(list(merged_long_extreme['Ticker']))
                #self.extreme_ticker_list = list(set(self.extreme_ticker_list))
                #print(self.date.date(), round(equity))
                #print(merged_long_extreme)


        short_positions_df = positions_df.loc[positions_df['Shares'] < 0]
        if not short_positions_df.empty:
            merged_short = pd.merge(short_positions_df[['Ticker', 'Shares', 'Entry']],
                                    prices_for_date[['Ticker', 'Close_Price']], on='Ticker', how='inner')
            merged_short = merged_short.dropna()
            merged_short['Product'] = merged_short['Close_Price'] * abs(merged_short['Shares'])
            merged_short['Unrealized PNL'] = (merged_short['Entry'] - merged_short['Close_Price']) * abs(merged_short['Shares'])
            unrealized_short = merged_short['Unrealized PNL'].sum()
            equity += unrealized_short + self.margin_account
            self.unrealized_pnl += unrealized_short
            #merged_short_extreme = merged_short[(merged_short['Entry'] / merged_short['Close_Price'] - 1) * 100 > 500]
            #if not merged_short_extreme.empty:
                #self.extreme_ticker_list.extend(list(merged_short_extreme['Ticker']))
                #self.extreme_ticker_list = list(set(self.extreme_ticker_list))
                #print(self.date.date(), round(equity))
                #print(merged_short_extreme)


        if (equity != 0 and equity != self.margin_account) or self.date == self.data['Date'].iloc[-1]:
            self.equity = equity

    def calculate_bm_equity_value(self):
        price_for_date = spy.loc[spy['Date'] == self.date, 'Close_Price'].iloc[0]
        self.equity_bm = price_for_date * self.spy_shares

    def update_portfolio(self):
        # calculate current value of portfolio and benchmark
        self.calculate_equity_value()
        if self.date in spy['Date'].values and self.date != initial_date:
            self.calculate_bm_equity_value()
        total_pf = self.equity + self.cash
        total_bm = self.equity_bm + self.cash_bm
        self.total_value_bm = total_bm
        self.total_pnl_bm = total_bm - self.init_cash


        # find date of last row
        last_row_date = self.portfolio['Date'].iloc[-1] if not self.portfolio.empty else None
        # update the last row or add new row
        if last_row_date == self.date:
            # Update the corresponding entry
            self.portfolio.iloc[-1] = pd.Series([self.date, self.cash, self.equity, total_pf, total_bm])
        else:
            new_row = pd.DataFrame(
                [[self.date, self.cash, self.equity, total_pf, total_bm]],
                columns=['Date', 'Cash', 'Equity', 'Total PF', 'Total BM'])
            self.portfolio = pd.concat([self.portfolio, new_row], ignore_index=True)

    def open_benchmark(self):
        self.cash_bm = self.init_cash
        self.date = initial_date
        first_spy_price = spy.loc[spy['Date'] == initial_date.strftime("%Y-%m-%d")]['Close_Price'].iloc[0]
        self.spy_shares = self.cash_bm / (
                    first_spy_price + ib_comission + ib_comission * pass_through_factor + third_party_fees)

        self.cash_bm -= self.get_transaction_cost(self.spy_shares, self.spy_shares * first_spy_price, "Open Long")
        self.transaction_cost_bm += self.get_transaction_cost(self.spy_shares, self.spy_shares * first_spy_price,
                                                              "Open Long")

        self.cash_bm -= self.spy_shares * first_spy_price
        self.equity_bm = self.spy_shares * first_spy_price
        self.trade_volume_bm += self.spy_shares * first_spy_price
        self.update_portfolio()

    def backtest(self):
        self.open_benchmark()
        print(f"\nBacktesting surprises from {initial_date.strftime('%Y')} to {last_date.strftime('%Y')}:")


        # Apply the date adjustments
        self.data['Date_Short'] = self.data['Date'].apply(lambda x: adjust_to_weekday(x + timedelta(days=days_short)))
        self.data['Date_Long'] = self.data['Date'].apply(lambda x: adjust_to_weekday(x + timedelta(days=days_long)))
        # Merge for 'Price_Short'
        self.data = pd.merge(self.data, prices[['Date', 'Ticker', 'Close_Price']], how='left', left_on=['Date_Short', 'Ticker'],
                        right_on=['Date', 'Ticker'])
        self.data = self.data.rename(columns={'Close_Price_x': 'Close_Price', 'Close_Price_y': 'Price_Short'})
        self.data = self.data.drop(columns=['Date_y'])
        # Merge for 'Price_Long'
        self.data = pd.merge(self.data, prices[['Date', 'Ticker', 'Close_Price']], how='left', left_on=['Date_Long', 'Ticker'],
                        right_on=['Date', 'Ticker'])
        self.data = self.data.drop(columns=['Date'])
        self.data = self.data.rename(columns={'Close_Price_x': 'Close_Price', 'Close_Price_y': 'Price_Long', 'Date_x': 'Date'})


        datadict = self.data.to_dict(orient='records')
        for i, dictrow in enumerate(datadict):
            # progress bar
            n = len(self.data)
            j = (i + 1) / n
            sys.stdout.write('\r')
            sys.stdout.write("[%-38s] %.1f%% " % ('#' * int(38 * j), 100 * j))
            sys.stdout.flush()

            self.date = dictrow['Date']

            if not i == len(self.data) - 1:
                next_date = datadict[i + 1]['Date']
            else:
                next_date = None
            if len(self.close_stack) > 0:
                to_close = self.close_stack.loc[self.close_stack['Date'] <= self.date]
                if not to_close.empty:
                    for close in to_close.itertuples():
                        shares = self.positions[close.Ticker]['Shares']
                        if shares > 0.0001 and self.cash >= self.get_transaction_cost(shares, shares * close.Price, "Close Long"):
                            self.execute_order("Close Long" , close.Ticker, close.Price, abs(shares), shares)
                        elif shares < -0.0001:
                            profit_loss = (self.positions[close.Ticker]['Entry'] - close.Price) * abs(shares)
                            cash_in = margin * self.positions[close.Ticker]['Entry'] * abs(shares) + profit_loss
                            if 0 <= (self.cash + cash_in - self.get_transaction_cost(shares, shares * close.Price, "Close Short")):
                                self.execute_order("Close Short", close.Ticker, close.Price, abs(shares), shares)
                    self.close_stack = self.close_stack.drop(self.close_stack.loc[self.close_stack['Date'] <= self.date].index)

            # if no annuncement on this date just update portfolio and benchmark
            if pd.isna(dictrow['Ticker']):
                if (next_date != self.date and next_date) or self.date == self.data['Date'].iloc[-1]:
                    self.update_portfolio()

                continue
            if pd.isna(dictrow['Close_Price']):
                if next_date != self.date and next_date:
                    self.update_portfolio()
                continue
            # extract variables from dictinonary
            price = dictrow['Close_Price']
            ticker = dictrow['Ticker']
            signal = dictrow['Signal']
            sue = dictrow['SUE']
            current_position = self.positions[ticker]['Shares']

            order_value = 5649
            # order_value = 42315 #tech parameter

            order_size = order_value / price
            # print(self.date.date(), round(self.cash), round(self.equity), round(self.portfolio['Total PF'].iloc[-1]))
            if signal == "No Surprise":
                if abs(current_position) <= 0.0001:
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                elif current_position > 0.0001:
                    signal = 'Close Long'
                    self.execute_order(signal, ticker, price, current_position, current_position)
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                elif current_position < -0.0001:
                    signal = 'Close Short'
                    profit_loss = (self.positions[ticker]['Entry'] - price) * abs(current_position)
                    cash_in = margin * self.positions[ticker]['Entry'] * abs(current_position) + profit_loss
                    if 0 <= (self.cash + cash_in - self.get_transaction_cost(order_size, order_value, signal)):
                        self.execute_order(signal, ticker, price, abs(current_position), current_position)
                        if next_date != self.date and next_date:
                            self.update_portfolio()
                        continue

            elif signal == 'Buy':
                if current_position > 0.0001:
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                elif current_position < -0.0001:
                    profit_loss = (self.positions[ticker]['Entry'] - price) * abs(current_position)
                    cash_in = margin * self.positions[ticker]['Entry'] * abs(current_position) + profit_loss
                    if 0 <= (self.cash + cash_in - self.get_transaction_cost(order_size, order_value, "Close Short")):
                        signal = "Close Short"
                        self.execute_order(signal, ticker, price, abs(current_position), current_position)
                if self.cash >= order_value + self.get_transaction_cost(order_size, order_value, "Open Long"):
                    signal = "Open Long"
                    if pd.isna(dictrow['Price_Long']):
                        if self.date >= (self.prices['Date'].iloc[-1] - timedelta(self.days_long)):
                            last_prices = self.prices.drop_duplicates(subset='Ticker', keep='last')
                            close_price = last_prices.loc[last_prices['Ticker'] == ticker, 'Close_Price'].iloc[0]
                        else:
                            if next_date != self.date and next_date:
                                self.update_portfolio()
                            continue
                    else:
                        close_price = dictrow['Price_Long']
                    self.execute_order(signal, ticker, price, order_size, current_position)
                    self.add_close_date("Close Long", ticker, close_price)
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                else:
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue

            elif signal == 'Sell':
                if current_position < -0.0001:
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                elif current_position > 0.0001:
                    signal = "Close Long"
                    self.execute_order(signal, ticker, price, current_position, current_position)
                if self.cash >= margin * order_value + self.get_transaction_cost(order_size, order_value, "Open Short"):
                    signal = "Open Short"
                    if pd.isna(dictrow['Price_Short']):
                        if self.date >= (self.data.iloc[-1, 0] - timedelta(self.days_short)):
                            last_prices = self.prices.drop_duplicates(subset='Ticker', keep='last')
                            close_price = last_prices.loc[last_prices['Ticker'] == ticker, 'Close_Price'].iloc[0]
                        else:
                            if next_date != self.date and next_date:
                                self.update_portfolio()
                            continue
                    else:
                        close_price = dictrow['Price_Short']
                    self.execute_order(signal, ticker, price, order_size, current_position)
                    self.add_close_date("Close Short", ticker, close_price)
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
                else:
                    if next_date != self.date and next_date:
                        self.update_portfolio()
                    continue
            if next_date != self.date and next_date:
                self.update_portfolio()
            # print(self.date.date(), round(self.cash), round(self.equity), " ==> Total: ", round(self.equity + self.cash))
            # if self.cash < 0:
                # print(self.date.date(), ticker, signal, round(sue, 4), round(price, 2), round(order_value), round(self.cash), round(self.equity))
        print(self.extreme_ticker_list)

        days = len(self.portfolio)
        years = days / 252

        #rf_rate_yearly = self.treasury_3m.groupby(self.treasury_3m.Date.dt.year)['Rate'].mean()
        risk_free = self.treasury_3m['Rate'].mean()


        # calculating daily log returns and total cumulative return for portfolio and benchmark
        self.portfolio['pf_daily'] = self.portfolio['Total PF'] / self.portfolio['Total PF'].shift(1)-1
        self.portfolio['bm_daily'] = self.portfolio['Total BM'] / self.portfolio['Total BM'].shift(1)-1
        pf_return_daily = self.portfolio['pf_daily'].dropna()
        pf_return_total = (self.cash + self.equity) / self.init_cash - 1
        pf_return_total_log = np.log(pf_return_total + 1)
        pf_return_ann_log = pf_return_total_log / years
        pf_return_ann = np.exp(pf_return_ann_log) - 1

        # calculating daily log returns of the benchmark
        bm_return_daily = self.portfolio['bm_daily'].dropna()
        bm_return_total=(self.cash_bm+self.equity_bm) / self.init_cash-1
        bm_return_total_log=np.log(bm_return_total+1)
        bm_return_ann_log = bm_return_total_log / years
        bm_return_ann = np.exp(bm_return_ann_log) - 1

        # calculating daily and annualized volatility for portfolio and benchmark
        pf_vola_daily = np.log(pf_return_daily+1).std()
        pf_vola_ann = pf_vola_daily * np.sqrt(252)
        bm_vola_daily = np.log(bm_return_daily+1).std()
        bm_vola_ann = bm_vola_daily * np.sqrt(252)

        # calculating portfolio beta
        returns_df = pd.DataFrame({'Portfolio': self.portfolio['pf_daily'].dropna(), 'Market': self.portfolio['bm_daily'].dropna()})
        cov_matrix = returns_df.cov()
        cov_pf_bm = cov_matrix.loc['Portfolio', 'Market']
        variance_bm = cov_matrix.loc['Market', 'Market']
        pf_beta = cov_pf_bm / variance_bm


        # calculating Jensen's Alpha
        jensens_alpha = pf_return_ann - risk_free - pf_beta * (bm_return_ann - risk_free)

        # calculating sharpe ratio to adjust returns for risk
        excess_return_pf = pd.merge(self.portfolio[['Date', 'pf_daily']], treasury_3m, how='inner', on=['Date'])
        excess_return_pf['rf_daily'] = (1 + excess_return_pf['Rate']) ** (1 / 365) - 1
        excess_return_pf['er_daily'] = excess_return_pf['pf_daily'] - excess_return_pf['rf_daily']
        er_daily = excess_return_pf['er_daily'].dropna()
        er_daily_log = np.log(er_daily + 1)
        excess_return_total_log = sum(er_daily_log)
        excess_return_ann_log = excess_return_total_log / years
        excess_return_ann = np.exp(excess_return_ann_log) - 1

        # calculating sharpe ratio of the benchmark
        excess_return_bm = pd.merge(self.portfolio[['Date', 'bm_daily']], treasury_3m, how='inner', on=['Date'])
        excess_return_bm['er_daily_bm'] = excess_return_bm['bm_daily'] - excess_return_pf['rf_daily']
        er_daily_bm = excess_return_bm['er_daily_bm'].dropna()
        excess_return_total_bm = sum(er_daily_bm)
        excess_return_ann_bm = excess_return_total_bm / years
        excess_return_ann_bm = np.exp(excess_return_ann_bm) - 1

        er_vola_bm = excess_return_bm['er_daily_bm'].std()
        er_vola_ann_bm = er_vola_bm * np.sqrt(252)

        pf_sharpe = np.mean(er_daily)/er_daily.std()
        bm_sharpe = np.mean(er_daily_bm) / er_daily_bm.std()


        # selling all benchmark etf shares
        print("\nBacktest complete!")
        self.cash_bm -= self.get_transaction_cost(self.spy_shares, self.equity_bm, "Close Long")
        self.transaction_cost_bm += self.get_transaction_cost(self.spy_shares, self.equity_bm, "Close Long")
        self.cash_bm = self.equity_bm
        self.trade_volume_bm += self.equity_bm
        self.equity_bm = 0

        # printing results
        lower_SUE, upper_SUE = self.sue_thresholds
        pf_return_daily *= 100
        bm_return_daily *= 100
        # print("\n========================================== Done with all surprises ==========================================\n")
        print(self.portfolio.round(2))
        # self.portfolio.to_csv('data/portfolio.csv', encoding='utf-8')

        print(f"\nResults for surprises with SUEs under {lower_SUE} and over {upper_SUE}:")
        print(f"Initial date: {initial_date.strftime('%Y-%m-%d')}. End date: {last_date.strftime('%Y-%m-%d')}:")
        print("=============================================================================================================")
        print(f"  Total return portfolio: {round(pf_return_total * 100)}% (${round(self.cash + self.equity - self.init_cash)}) vs. "
              f"benchmark: {round(bm_return_total * 100)}% (${round(self.cash_bm - self.init_cash)}) => Total excess return: {round((pf_return_total - bm_return_total) * 100)}%")
        print(f"  Annualized return portfolio: {round(pf_return_ann * 100, 2)}% vs. "
              f"benchmark: {round(bm_return_ann * 100, 2)}% => Annualized excess return: {round(pf_return_ann * 100 - bm_return_ann * 100, 2)}%")
        print("_____________________________________________________________________________________________________________")
        print(f"  Annualized volatility portfolio: {round(pf_vola_ann * 100, 2)}% vs. benchmark: {round(bm_vola_ann * 100, 2)}%")
        print("_____________________________________________________________________________________________________________")
        print(f"  Portfolio beta: {round(pf_beta, 4)}")
        print(f"  Jensen's Alpha: {round(jensens_alpha * 100, 2)}%")
        print("_____________________________________________________________________________________________________________")
        print(f"  Sharpe Ratio portfolio: {round(pf_sharpe, 4)} vs. benchmark : {round(bm_sharpe, 4)}")
        print(f"  Risk-free rate average of historical 3-month US treasury bonds: {round(risk_free * 100, 2)}%")
        print("_____________________________________________________________________________________________________________")
        print(f"  Total transaction cost portfolio: ${round(self.transaction_cost)} (Transactions: {self.counter}, Shares traded: {round(self.traded_shares)}, Volume: ${round(self.trade_volume)}) \n"
              f"  Total transaction cost benchmark: ${round(self.transaction_cost_bm)} (Transactions: 2, Shares traded: {2 * round(self.spy_shares)}, Trade volume: ${round(self.trade_volume_bm)})")
        print("_____________________________________________________________________________________________________________")
        print(f"  order_size: {order_value}, sue_long: {upper_threshold}, sue_short: {lower_threshold}, days_long: {days_long}, days_short: {days_short}")
        print("\n")

        summary_text = f"""\
Period: {initial_date.date()} to {last_date.date()}

SUE-Score thresholds: {lower_threshold} to {upper_threshold}
A long position is opened if the SUE-score exceeds the upper threshold,
and a short position if it falls below the lower threshold.

Total return portfolio: {round(pf_return_total * 100)}% (${round(self.cash + self.equity - self.init_cash)})
Total return benchmark: {round(bm_return_total * 100)}% (${round(self.cash_bm - self.init_cash)})

Annualized return portfolio: {round(pf_return_ann * 100, 2)}%
Annualized return benchmark: {round(bm_return_ann * 100, 2)}%
Annualized excess return: {round(pf_return_ann * 100 - bm_return_ann * 100, 2)}%

Volatility portfolio: {round(pf_vola_ann * 100, 2)}%
Volatility benchmark: {round(bm_vola_ann * 100, 2)}%

Sharpe Ratio portfolio: {round(pf_sharpe, 4)}
Sharpe Ratio benchmark: {round(bm_sharpe, 4)}
Jensen's Alpha portfolio: {round(jensens_alpha * 100, 2)}%
Risk-free rate (3M average): {round(risk_free * 100, 2)}%

Transaction cost portfolio: ${round(self.transaction_cost)} on ${round(self.trade_volume)} volume
Transaction cost benchmark: ${round(self.transaction_cost_bm)} on ${round(self.trade_volume_bm)} volume
"""
        self.save_backtest_summary(summary_text)


    def save_backtest_summary(self, summary_text, filename="output/backtest_result.txt"):
        import os
        os.makedirs("output", exist_ok=True)

        with open(filename, "w") as f:
            f.write("=== PEAD-Strategy Backtest Results ===\n\n")
            f.write(summary_text.strip())

        import subprocess
        subprocess.run(["open", filename])

    def plot_results(self):
        self.portfolio.set_index('Date')
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 12))

        pf_value = self.portfolio['Total PF']
        ax1.plot(self.portfolio['Date'], pf_value, label='Portfolio Value', color='green')
        bm_value = self.portfolio['Total BM']
        ax1.plot(self.portfolio['Date'], bm_value, label='Total BM', color='red')

        ax1.legend(loc='upper left')
        ax1.set_title(f"\nPEAD-Strategy vs. S&P 500: Backtest {initial_date.strftime('%Y')} - {last_date.strftime('%Y')}")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('USD')
        ax1.fmt_xdata = mdates.DateFormatter('%d.%m.%Y')
        # ax1.set_title('Earnings Announcement Strategy vs. S&P 500: Backtest 1995 - 2022', loc='left', y=0.85, x=0.02, fontsize='medium')
        # Text in the x-axis will be displayed in 'YYYY-mm' format.
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        # Rotates and right-aligns the x labels so they don't crowd each other.
        for label in ax1.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        ax2.plot(self.portfolio['Date'], self.portfolio['Equity'], label='Equity Value', color='orange')
        ax2.set_ylabel('Equity Value')

        ax2.plot(self.portfolio['Date'], self.portfolio['Cash'], label='Cash', color='purple')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('USD')
        ax2.legend(loc='upper left')
        # ax3.set_title('Cash and Equity Balances of the Long-Short Portfolio')
        ax2.fmt_xdata = mdates.DateFormatter('%d.%m.%Y')

        # ax2.set_title('Cash and Equity Balances of the Long-Short Portfolio', loc='left', y=0.85, x=0.02, fontsize='medium')
        # Text in the x-axis will be displayed in 'YYYY-mm' format.
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        # Rotates and right-aligns the x labels so they don't crowd each other.
        for label in ax2.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        # Rotate and align the tick labels so they look better.
        # fig.autofmt_xdate()
        lower_SUE, upper_SUE = self.sue_thresholds
        plt.savefig(f"output/plot_{initial_date.strftime('%Y')}-{last_date.strftime('%Y')}.png",
                    bbox_inches='tight')
        plt.show()



days_long = 85
days_short = 9

# tech parameters
# days_long = 85
# days_short = 10


initial_date = pd.to_datetime('2020-12-31')
last_date = pd.to_datetime('2023-12-31')

# initial_date = pd.to_datetime('1994-01-03')
# last_date = pd.to_datetime('2008-12-31')

# import and prepare daily prices data
prices = pd.read_csv('data/myprices.csv')
prices['Date'] = pd.to_datetime(prices['Date'])
prices['Ticker'] = prices['Ticker'].astype('string')
prices = prices.rename(columns={'Date': 'Date', 'Close_Price': 'Close_Price', 'Ticker': 'Ticker'})
prices.sort_values(by=['Ticker', 'Date'], inplace=True)

# for hypothesis filter for 2 tech stocks only ------------------------------------------------------------------------
# prices = prices.loc[(prices['SIC'] == 737) | (round(prices['SIC']/10) == 35)]

# List of tickers to exclude
exclude_tickers = ['CHTR']

prices = prices[~prices['Ticker'].isin(exclude_tickers)]
prices = prices.loc[prices["Close_Price"] > 0]

# import and prepare surprises data from Surprises_Final.csv
data = pd.read_csv("data/surprises_constituents.csv", dtype='unicode')
data = data.rename(columns={"anndats": "Date", "oftic": "Ticker", "suescore": "SUE"})
data['Ticker'] = data['Ticker'].astype('string')
data['Date'] = pd.to_datetime(data['Date'])
data['SUE'] = data['SUE'].astype('float64')
data['surpmean'] = data['surpmean'].astype('float64')
data['surpstdev'] = data['surpstdev'].astype('float64')
data['actual'] = data['actual'].astype('float64')
data = data.sort_values(by=['Date', 'Ticker'])

# fix missing standard deviations by taking the mean and calculating the missing SUE values
average_surpstdev = data['surpstdev'].mean()
data['surpstdev'].fillna(average_surpstdev, inplace=True)
data['surpstdev'] = data['surpstdev'].mask(data['surpstdev'] == 0, average_surpstdev)
data['SUE'] = data['SUE'].fillna((data['actual'] - data['surpmean']) / data['surpstdev'])
data = data.dropna()

columns_titles = ["Date", "Ticker", "SUE"]
data = data.reindex(columns=columns_titles)

# ensuring surprise dates are within the desired time peridod
mask = (data['Date'] >= initial_date) & (data['Date'] <= last_date)
data = data.loc[mask]

# exclude unadjusted tickers
data = data[~data['Ticker'].isin(exclude_tickers)]

# adding dates with no earnings announcement

days = (last_date - initial_date).days
timeline = pd.Series([initial_date + timedelta(days=x) for x in range(days+1)])
data = pd.merge(timeline.rename('Date'), data, how='outer', on='Date')
data['Date'] = pd.to_datetime(data['Date'])

# Removing all week-end dates
data = data.loc[data['Date'].dt.weekday <= 4]
data = pd.merge(data, prices[['Date', 'Ticker', 'Close_Price']], on=['Date', 'Ticker'], how='left')

# import and prepare benchmark data (S&P 500 index fund) from SPY.csv
spy_file = 'data/spy.csv'
usecols = ["Date", "Close_Price"]
spy = pd.read_csv(spy_file, sep=",", usecols=usecols, dtype='unicode')
spy = spy.rename(columns={'Adj Close': 'Close_Price'})
spy['Date'] = pd.to_datetime(spy['Date'])
spy['Close_Price'] = spy['Close_Price'].astype('float64')
spy = spy.fillna(0)

# reading treasury yield data
treasury_3m = pd.read_csv('data/treasury.csv')[['Date', '3 Mo']]
treasury_3m = treasury_3m.rename(columns={'3 Mo': 'Rate'})
treasury_3m['Date'] = pd.to_datetime(treasury_3m['Date'], format='%m/%d/%y')
# treasury_3m['Date'] = treasury_3m['Date'].dt.strftime('%Y-%m-%d')
treasury_3m['Rate'] /= 100
treasury_3m = treasury_3m.dropna()

# Create a mask based on the date range in prices
mask = ((treasury_3m['Date'] >= prices['Date'].min()) & (treasury_3m['Date'] <= prices['Date'].max()))
treasury_3m = treasury_3m.loc[mask]
treasury_3m = treasury_3m.sort_values('Date')

bidask = pd.read_csv('data/bidask.csv')
bidask = bidask.drop(columns={'PERMNO'})
bidask = bidask.dropna()
bidask['spread'] = (bidask['ASK'] - bidask['BID']) / (bidask['ASK'] + bidask['BID'] / 2)
bidask = bidask.loc[bidask['spread'] > 0]
spread_median = bidask['spread'].median()

margin = 0.5

ib_comission = 0.0035
min_ib_commission = 0.35
third_party_fees = 0.0002 + 0.003
pass_through_factor = 0.000175 + 0.00056
finra_selling_fee = 0.000166
stock_borrowing_cost = 0.0025
sec_selling_fee = 0.000008

unique_tickers = data['Ticker'].unique().dropna()
positions = {ticker: {'Shares': 0, 'Entry': None, 'Exit': None} for ticker in unique_tickers}
portfolio = pd.DataFrame(columns=['Date', 'Cash', 'Equity', 'Total PF', 'Total BM'])
portfolio = portfolio.astype('float64')
portfolio['Date'] = portfolio['Date'].astype('datetime64[ns]')


close_stack = pd.DataFrame({c: pd.Series(dtype=t) for c, t in {'Date': 'datetime64[ns]', 'Ticker': 'int', 'Shares': 'float', 'Signal': 'int'}.items()})
close_stack['Ticker'] = close_stack['Ticker'].astype("string")
close_stack['Signal'] = close_stack['Signal'].astype("string")

# Define the function to adjust dates to the next weekday
def adjust_to_weekday(d):
    weekday = d.weekday()
    if weekday > 4:  # Adjust if it's Saturday or Sunday
        return d + timedelta(days=7 - weekday)
    return d

# data.to_csv('data/data.csv', encoding='utf-8', index=False)
# prices.to_csv('data/myprices.csv', encoding='utf-8', index=False)

print("\nSuccessfully retrieved daily prices and analyst data")

lower_threshold = -3.78
upper_threshold = 0.635

#tech parameters
# lower_threshold = -0.65
# upper_threshold = 0.15

data.loc[(lower_threshold < data['SUE']) & (upper_threshold > data['SUE']), 'Signal'] = "No Surprise"
data.loc[(lower_threshold > data['SUE']), 'Signal'] = "Sell"
data.loc[(upper_threshold <= data['SUE']), 'Signal'] = "Buy"
data['Signal'] = data['Signal'].astype('string')
thresholds = (lower_threshold, upper_threshold)
backtester = Backtester(data, prices, spy, treasury_3m, thresholds, days_long, days_short)
backtester.backtest()
backtester.plot_results()


