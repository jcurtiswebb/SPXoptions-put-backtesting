# %% [code]
# %% [code]
 # %% [code]
# %% [code]
# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from textwrap import wrap
from time import perf_counter
from datetime import datetime, timedelta, time
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

########
# Version and helpers
########
spx_strat_v = 0

def study_name_reducer(i_name):
    i_name = i_name.replace("Delta","D")
    i_name = i_name.replace("Option","O")
    i_name = i_name.replace("Selector","S")
    i_name = i_name.replace("Static", "S")
    i_name = i_name.replace("Policy", "P")
    i_name = i_name.replace("HoldToExp","HTE")
    return i_name

# The following code builds the fully described study, including entry and exit policy, data, trial search params, etc
def get_study_name(op_sel, ent_pol_study, study_param_ranges, 
                   configs_ent_pol, ex_pol_study, configs_ex_pol,
                   configs_strat, configs, df_study, spx_strat_v,
                   notebook_name, notebook_v
                  ):
    """
    builds the fully described study as a string, including entry and exit policy, data, trial search params, etc

    :param op_sel: The class of the option selector being used | None
    :param ent_pol_study: The class of the entry policy being used | None
    :param study_param_ranges: A dictionary of study parameters being used and their ranges as tuples | None
    :param configs_ent_pol: A dictionary of parameters of the entry policy, not including those included in param ranges | None
    :param ex_pol_study: The class of the exit policy being used | None
    :param configs_ex_pol: A dictionary of parameters of the exit policy, not including those included in param ranges | None
    :param configs_strat: A dictionary of parameters of the strategy, not including those included in param ranges | None
    :param configs: A dictionary of parameters of other configuration items, not including those included in param ranges | None
    :param df_study: the dataframe used for the study
    :param spx_strat_v: The version of spx_strategies
    :param notebook_name: The name of the calling notebook
    :param notebook_v: The version of the calling notebook
    :return: study: string
    """
    study = ''
    
    study += op_sel.__name__ + '_' if op_sel is not None else ''
    
    study += ent_pol_study.__name__ + '_' if ent_pol_study is not None else ''
    

    if study_param_ranges is not None:
        # concatenate ranges first
        for k,v in study_param_ranges.items():
            study+="_"+k+":"+str(v[0])+"_"+str(v[1])

    if configs_ent_pol is not None:
        # concatenate static values second
        for k,v in configs_ent_pol.items():
            study += f"_{k}_{v}"

    study += f"_{ex_pol_study.__name__}" if ex_pol_study is not None else ''

    if configs_ex_pol is not None:
        for k,v in configs_ex_pol.items():
            study += f"_{k}_{v}"

    if configs_strat is not None:
        for k,v in configs_strat.items():
            study += f"_{k}_{v}"
        
    if configs is not None:
        for k,v in configs.items():
            study += f"_{k}_{v}"

    study += "_"+ df_study['quote_date'].min().date().strftime("%y%m%d") +'_' + df_study['quote_date'].max().date().strftime("%y%m%d")

    study += "_spx_st_" + str(spx_strat_v)

    study += "_" + notebook_name + "_" + str(notebook_v)

    study = study_name_reducer(study)
    return study


#########
# Abstract Strategy & Abstract Policy
#########

class AbstractStrategy(ABC):
    def __init__(self, entry_policy, exit_policy, df_ty, initial_portfolio_value, commission = 1.5, debug = False):
        self.entry_policy = entry_policy
        self.exit_policy = exit_policy
        self.commission = commission
        self.initial_portfolio_value = initial_portfolio_value
        self.df_ty = df_ty
        self.debug = debug
        self.max_bet_scaling = 0.02
        self.df = None
        self.df_trades_implausible = None
        self.remove_implausible_trades = False
        self.fix_unrealistic_trades = True
        self.fix_threshold = 10
        
        # If another method overrode df_trades, we will respect it.
        if hasattr(self, 'df_trades') == False:
            self.df_trades = pd.DataFrame(columns=['trade_date', 'expiration', 'trade_count', 'collected', 'lost_c', 'lost_p'])
       
    def performCalcs(self):
        if self.df is None:
            raise RuntimeError("self.df is not set, please manually set it before running performCalcs.")
        
        df = self.df
        exp_dates = self.df['expiration'].unique()
        df_closing = df[(df['quote_date'].isin(exp_dates))&(df['quote_time']=='16:00:00')].groupby(['quote_date'])['price'].mean()
        
        df_trades = self.df_trades
        df_trades.drop(columns=['price'], inplace=True, errors='ignore')
        df_trades = df_trades.merge(df_closing.to_frame(), left_on=['expiration'], right_index=True)
        
        #prior to commission calculation, we reduce # trade_count by number of no-trades
        
        df_trades['commission'] = df_trades['trade_count']*self.commission
        df_trades['lost'] = df_trades['lost_c'] + df_trades['lost_p']
        df_trades['net'] = df_trades['collected'] - df_trades['lost'] - df_trades['commission']
        df_trades['portfolio_value'] = 0.0
        #df_trades['daily_return'] = 0.0
        #df_trades['margin_utilization']=0.0
        initial_portfolio_value = self.initial_portfolio_value
        df_trades['net_cumsum']=df_trades['net'].cumsum()

        df_trades['portfolio_value'] = initial_portfolio_value + df_trades['net_cumsum']
        df_trades['transaction_return'] = df_trades['net'] / df_trades['portfolio_value'].shift(1)
        net = df_trades['net'].iloc[0]
        df_trades.loc[0,'transaction_return'] = df_trades['net'].iloc[0] / initial_portfolio_value
        df_trades['cum_return'] = (df_trades['portfolio_value'] - initial_portfolio_value) / initial_portfolio_value

        df_ty = self.df_ty.copy()
        df_ty['daily_risk_free_return'] = (df_ty['Adj Close'] / 252) / 100
        df_ty.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True, errors='ignore')

        df_trades.drop(['daily_risk_free_return'],axis=1,inplace=True, errors='ignore')
        df_trades = pd.merge(df_trades, df_ty, left_on='trade_date', right_on='Date')
        df_trades.drop(['Date'],axis=1,inplace=True)
        df_trades.fillna(0, inplace=True)

        

        trade_count = df_trades.shape[0]
        win_count = df_trades[df_trades['net']>0].shape[0]
        loss_count = df_trades[df_trades['net']<0].shape[0]
        std_trans_return = df_trades['transaction_return'].std()
        std_trans_return_less_rf = (df_trades['transaction_return'] - df_trades['daily_risk_free_return']).std()
        
        sc_cols = [col for col in df_trades.columns.to_list() if 'strike_sc' in col]
        lc_cols = [col for col in df_trades.columns.to_list() if 'strike_lc' in col]
        sp_cols = [col for col in df_trades.columns.to_list() if 'strike_sp' in col]
        lp_cols = [col for col in df_trades.columns.to_list() if 'strike_lp' in col]
        
        self.df_trades_dropped = df_trades[df_trades['trade_count'] <= 0.0].copy()
        print(f"Dropped {self.df_trades_dropped.shape[0]} records because they had no trades on that day. They are available in df_trades_dropped")
        df_trades.drop(df_trades[df_trades['trade_count'] <= 0.0].index, inplace=True)

        for col in df_trades.columns:
            df_trades['short_long_balance_call'] = df_trades[lc_cols].gt(0).sum(axis=1) - df_trades[sc_cols].gt(0).sum(axis=1)
            df_trades['short_long_balance_put'] = df_trades[lp_cols].gt(0).sum(axis=1) - df_trades[sp_cols].gt(0).sum(axis=1)
            
        if (df_trades['short_long_balance_call'] == 0).all() and (df_trades['short_long_balance_put'] == 0).all():
            df_trades.drop(['short_long_balance_call','short_long_balance_put'],axis=1,inplace=True)
            # These are spreads we need to calculate: max loss, return on max risk, std deviation of return on max risk, and risk adjusted return on max risk
            df_trades['gross_max_loss'] = df_trades.apply(lambda row : self.get_max_loss(row, sc_cols, lc_cols, sp_cols, lp_cols), axis=1)
            df_trades['net_max_loss'] = df_trades['gross_max_loss'] - df_trades['collected']
            df_trades['return_on_max_risk'] = df_trades['net'] / df_trades['net_max_loss']
            df_trades['return_on_max_risk'] = df_trades['return_on_max_risk'].fillna(0)
            # TODO : can we remove this intermediate calculation and do it in a one-liner
            df_trades['scaled_return_on_max_risk'] = df_trades['return_on_max_risk']*self.max_bet_scaling + 1
            
            
            
            if self.fix_unrealistic_trades:
                # TODO change collected instead of net_max_loss and recalculate
                # Fixing 
                
                self.df_trades_unrealistic = df_trades[(df_trades['net_max_loss']<=0)|(df_trades['gross_max_loss']>10*df_trades['net_max_loss'])].copy()
                print(f"Fixing {self.df_trades_unrealistic.shape[0]} trades. They are listed in the df_trades_unrealistic dataframe of the strategy object.")
                
                df_trades.loc[(df_trades['net_max_loss']<=0)|(df_trades['gross_max_loss']>self.fix_threshold*df_trades['net_max_loss']),'collected'] = df_trades['collected'].median()
                df_trades['net'] = df_trades['collected'] - df_trades['lost'] - df_trades['commission']
                df_trades['net_max_loss'] = df_trades['gross_max_loss'] - df_trades['collected']
                df_trades['return_on_max_risk'] = df_trades['net'] / df_trades['net_max_loss']
                df_trades['return_on_max_risk'] = df_trades['return_on_max_risk'].fillna(0)
                # TODO : can we remove this intermediate calculation and do it in a one-liner
                df_trades['scaled_return_on_max_risk'] = df_trades['return_on_max_risk']*self.max_bet_scaling + 1
            
            self.df_trades_implausible = df_trades[df_trades['return_on_max_risk'] > self.fix_threshold].copy()
            if self.remove_implausible_trades and self.df_trades_implausible.shape[0] > 0:
                print(f"Dropping {self.df_trades_implausible.shape[0]} trades. They are listed in the df_trades_implausible dataframe of the strategy object.")
                df_trades.drop(df_trades[df_trades['return_on_max_risk'] > 1.0].index, inplace=True)
            
            
        
        self.df_trades = df_trades
        dict_results = {}
        if 'net_max_loss' in df_trades.columns:
            dict_results['Cumulative Return'] = round(100*df_trades['scaled_return_on_max_risk'].cumprod().iloc[-1] - 100,3)
            dict_results['Max Drawdown'] = round(df_trades['cum_return'].min()*100,3)
            dict_results['Mean Net Max Loss'] = round(df_trades['net_max_loss'].mean(),3)
            dict_results['Max Net Max Loss'] = round(df_trades['net_max_loss'].max(),3)
            dict_results['Mean Return on Max Risk'] = round(df_trades['return_on_max_risk'].mean(),3)
            dict_results['Std Dev of Return on Max Risk'] = round(df_trades['return_on_max_risk'].std(),3)
            dict_results['Risk Adj Cumulative Return On Scaled Max Risk'] = round(dict_results['Cumulative Return']/dict_results['Std Dev of Return on Max Risk'],3)

        else:
            dict_results['Cumulative Return'] = round(df_trades['cum_return'].iloc[-1]*100,3)
            dict_results['Max Drawdown'] = round(df_trades['cum_return'].min()*100,3)
            dict_results['Mean Win Trans Return'] = round(df_trades[df_trades['transaction_return']>0]['transaction_return'].mean()*100,3)
            dict_results['Mean Loss Trans Return'] = round(df_trades[df_trades['transaction_return']<0]['transaction_return'].mean()*100,3)
            dict_results['Mean Trans Return'] = round(df_trades['transaction_return'].mean()*100,3)
            dict_results['Std Dev of Trans Return'] = round(df_trades['transaction_return'].std(),3)
            dict_results['Sharpe Ratio static STD'] = round(np.sqrt(252)*(df_trades['transaction_return'].mean()-df_trades['daily_risk_free_return'].mean())/std_trans_return,3)
            dict_results['Sharpe Ratio with RF STD'] = round(np.sqrt(252)*(df_trades['transaction_return'].mean()-df_trades['daily_risk_free_return'].mean())/std_trans_return_less_rf,3)
            dict_results['Risk Adj Cumulative Return'] = round(df_trades['cum_return'].iloc[-1]*100/std_trans_return,3)
            dict_results['Dampened Risk Adj Cumulative Return'] = round(df_trades['cum_return'].iloc[-1]*100/np.sqrt(std_trans_return),3)
            

        # these entries are the same regardless of scenario
        dict_results['Trading Days'] = trade_count
        dict_results['Wins'] = win_count
        dict_results['Losses'] = loss_count
        dict_results['Breakeven'] = df_trades[df_trades['net']==0.0].shape[0]
        dict_results['Win/Loss Ratio'] = round(win_count/trade_count*100,3)
        dict_results['Mean Win'] = round(df_trades[df_trades['net']>0]['net'].mean(),3)
        dict_results['Mean Loss'] = round(df_trades[df_trades['net']<0]['net'].mean(),3)
        dict_results['Mean Net Trans'] = round(df_trades['net'].mean(),3)
        dict_results['Std Dev of Net Trans'] = round(df_trades['net'].std(),3)
        dict_results['Max Loss'] = round(df_trades['net'].min(),3)
        dict_results['Max Win'] = round(df_trades['net'].max(),3)


        # We have two types of charts; naked option charts and spread charts.
        # Lets determine what set of charts to show here:
        if self.debug:
            if 'net_max_loss' in df_trades.columns:
                plt.style.use(plt.style.available[15])

                fig = plt.figure(figsize=(16,9))

                gs = fig.add_gridspec(2,2)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, :])

                fig.subplots_adjust(top=0.9)
                plt.subplots_adjust(hspace=0.5)
                fig.suptitle("\n".join(wrap(str(self),100)),fontweight=0.5,fontsize=20)

                df_trade_plot = df_trades.copy()
                df_trade_plot.set_index('expiration', inplace=True)
                df_trade_plot = df_trade_plot['scaled_return_on_max_risk'].cumprod()
                df_trade_plot.plot(ax=ax1)
                ax1.set_title("Scaled Return (Risking 2% of Portfolio Each Day)")
                ax1.set_ylabel('% Return')


                df_trades_transaction_return = df_trades.copy()
                df_trades_transaction_return['return_on_max_risk'] *= 100 
                df_trades_transaction_return.set_index('expiration', inplace=True)
                df_trades_transaction_return['return_on_max_risk'].plot(ax=ax2,linestyle='None', marker="o")
                ax2.set_title("Return on Max Risk (%)")

                sc_cols = [col for col in df_trades.columns.to_list() if 'strike_sc' in col]
                lc_cols = [col for col in df_trades.columns.to_list() if 'strike_lc' in col]
                sp_cols = [col for col in df_trades.columns.to_list() if 'strike_sp' in col]
                lp_cols = [col for col in df_trades.columns.to_list() if 'strike_lp' in col]
                strike_cols = sc_cols + lc_cols + sp_cols + lp_cols
                # strike_cols.append('price')
                df_trades_strike_plot = df_trades.copy()
                df_trades_strike_plot.set_index('expiration', inplace=True)
                df_trades_strike_plot = df_trades_strike_plot.loc[:,strike_cols]
                df_trades_strike_plot.plot(ax=ax3, linestyle='None')
                num_calls = len(sc_cols)
                num_puts = len(sp_cols)
                for i in range(num_calls):
                    ax3.fill_between(df_trades_strike_plot.index.to_list(),df_trades_strike_plot[sc_cols[i]], df_trades_strike_plot[lc_cols[i]], 
                                     where=(df_trades_strike_plot[lc_cols[i]] > df_trades_strike_plot[sc_cols[i]]),alpha=0.15, color='blue')

                for i in range(num_puts):
                    ax3.fill_between(df_trades_strike_plot.index.to_list(),df_trades_strike_plot[sp_cols[i]], df_trades_strike_plot[lp_cols[i]], 
                                     where=(df_trades_strike_plot[lp_cols[i]] < df_trades_strike_plot[sp_cols[i]]),alpha=0.15, color='green')

                df_trades_strike_plot = df_trades.copy()
                df_trades_strike_plot.set_index('expiration', inplace=True)
                df_trades_strike_plot = df_trades_strike_plot.loc[:,['price']]
                df_trades_strike_plot.plot(ax=ax3)
                
                # create manual symbols for legend
                call_patch = mpatches.Patch(color='blue', label='call spread (credit)')
                put_patch = mpatches.Patch(color='green', label='put spread (credit)')  
                spx_line = Line2D([0], [0], label='SPX close', color='orange')
                manual_legend = [call_patch, put_patch, spx_line]
                
                plt.legend(handles=manual_legend)
                ax3.set_title('Spread Region')
                
            else:
                # if you want to scale the chart, you should do it here
                df_trade_plot = df_trades.copy()
                df_trade_plot.set_index('expiration', inplace=True)
                df_trade_plot = df_trade_plot['net'].cumsum()


                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                secax = ax.secondary_yaxis('right', functions=(self.net2pct, self.pct2net))
                secax.set_ylabel('% return')
                df_trade_plot.plot(ax=ax)
                plt.title("\n".join(wrap(str(self),50)))
                plt.grid()
                plt.savefig(f"{str(self)}.png")
                if self.debug == False:
                    plt.close(fig)

                df_trades_transaction_return = df_trades.copy()
                df_trades_transaction_return['transaction_return'] *= 100 
                df_trades_transaction_return.set_index('expiration', inplace=True)
                fig = plt.figure()
                ax = df_trades_transaction_return['transaction_return'].plot(linestyle='None', marker="o")
                ax.set_ylabel('Transaction Return %')
                plt.title("\n".join(wrap(str(self),50)))
                plt.grid()
                plt.savefig(f"daily_ret_{str(self)}.png")
                if self.debug == False:
                    plt.close(fig)

        if self.debug:
            print("*****  BACKTEST RESULTS  ****")
            for key, value in dict_results.items():
                print(f"{key:<45}{value:>10}")
            print("\n")
            
        # TODO : Find best spot for df_trades ordering. Maybe during option selection?
        all_nonstrike_cols = [col for col in df_trades.columns.to_list() if 'strike_' not in col]
        strike_cols = [col for col in df_trades.columns.to_list() if 'strike_' in col]
        all_cols = all_nonstrike_cols[:3] + strike_cols + all_nonstrike_cols[3:]
        self.df_trades = df_trades[all_cols]
        
        return dict_results
    
    def get_max_loss(self, row, sc_cols, lc_cols, sp_cols, lp_cols):
        call_count = len(sc_cols)
        put_count = len(sp_cols)
        
        max_losses = np.zeros(max(call_count, put_count))
        
        for i in range(len(max_losses)):

            # for index i, does the put column exist?
            if f"strike_sp_{i}" in sp_cols:
                max_losses[i] = row[f"strike_sp_{i}"] - row[f"strike_lp_{i}"]
                
            if f"strike_sc_{i}" in sc_cols:
                max_losses[i] =  max((row[f"strike_lc_{i}"] - row[f"strike_sc_{i}"]), max_losses[i])
                
        
            
        return sum(max_losses) * 100

                
        
        
    def net2pct(self,x):
        return (x / self.initial_portfolio_value)*100

    def pct2net(self,x):
        return self.initial_portfolio_value * x/100
        
    @abstractmethod
    def evaluate(self,df):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
class AbstractPolicy(ABC):
    def getRoundedSlippedPrice(self,bid,ask, trans_type):
        mid = (bid + ask)/0.02
        
        magnitude = ask - bid
        second_slip = int(magnitude/20)*0.05
        
        if trans_type == 'sell':
            ask -= second_slip
        elif trans_type=='buy':
            bid += second_slip
        
        if mid % 1 != 0.0:
            # We need to slip the bid/ask spread
            if trans_type == 'sell':
                ask -= 0.05 + second_slip
            elif trans_type=='buy':
                bid += 0.05 + second_slip

        mid = round((bid + ask)/0.02,1)


        
        return mid
    
    def len_check(self,df, opt_type, strike, contract_date, quote_time):
        if len(df)==0:
            raise Exception(f"Fatal error. Option Type {opt_type} | Strike : {strike} not found for date : {contract_date} and time : {quote_time}")
    
    def get_contract_strike(self, curr_date, contract_date, target_delta, i_df, trans_type):
        # -1 means no delta should be selected
        if target_delta ==-1:
            return pd.Series([0,0,0])
#         df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date)].min()
        df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date)]
    
        if len(df1) == 0:
            target_delta = 0 if target_delta is None else target_delta
            print(f"WARNING : get_contract_strike no records found: {trans_type} Delta:{target_delta:.3f} {curr_date}, {contract_date}")
            return pd.Series([0,0,0])
        
        df1 = df1.iloc[0]
        rounded_price = self.getRoundedSlippedPrice(df1['bid'],df1['ask'],trans_type)

        
        return pd.Series([df1['strike'],df1['delta'],rounded_price])
    
    def get_amount_lost(self,df,row):
        contract_date = row['expiration']
        
        df1 = df[(df['quote_date']==contract_date) & (df['expiration']==contract_date) & (df['quote_time']=='16:00:00')]
        underlying = df1['price']

        if len(underlying)==0:
            raise Exception(f"Fatal error. Underlying price not found for date : {contract_date} and time : '16:00:00'")

        underlying = underlying.iloc[0]
        
        total_lost_c = 0.0
        total_lost_p = 0.0
        for col in row.index.values:
            if 'strike_sc' in col and row[col] != 0.0 and underlying > row[col]:
                    total_lost_c = total_lost_c + (underlying - row[col])*100
            if 'strike_lc' in col and row[col] != 0.0 and underlying > row[col]:
                    total_lost_c = total_lost_c - (underlying - row[col])*100
            if 'strike_sp' in col and row[col] != 0.0 and underlying < row[col]:
                    total_lost_p = total_lost_p + (row[col] - underlying)*100
            if 'strike_lp' in col and row[col] != 0.0 and underlying < row[col]:
                    total_lost_p = total_lost_p - (row[col] - underlying)*100
        return pd.Series([total_lost_c, total_lost_p])

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
    
#########
# AbstractEntryPolicy, AbstractStaticEntryPolicy, AbstractDynamicEntryPolicy
#########

class AbstractEntryPolicy(AbstractPolicy):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
class AbstractStaticEntryPolicy(AbstractEntryPolicy):
    @abstractmethod
    def populateTrades(self, data):
        pass
        
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class AbstractDynamicEntryPolicy(AbstractEntryPolicy):
    @abstractmethod
    def evaluateTradingCondition(self, eval_date, eval_time, df_trade_row, df_data):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
    
class OptionSelectorStaticEntryPolicy(AbstractStaticEntryPolicy):
    def __init__(self, dte, trade_time, option_selector):
        self.dte = dte
        self.trade_time = trade_time
        self.option_selector = option_selector
        super().__init__()
    
    def populateTrades(self, df):
        df_exp = df.loc[df[df['dte']==self.dte].groupby('expiration')['dte'].idxmin()]
        df_dates = df_exp.loc[df_exp.groupby('quote_date')['dte'].idxmin()]
        last_date = df_dates['quote_date'].max()
        df_dates = df_dates[df_dates['expiration']<=last_date]
        df_trades = df_dates[['quote_date','expiration']].copy()
        df_trades.rename({'quote_date': 'trade_date'}, axis='columns', inplace=True)
        df_trades['trade_count'] = 0
        
        df_data = df[(df['quote_time'] == self.trade_time) & (df['dte']==self.dte)]
        
        try:
            df_trades = self.option_selector.populateTrades(df_data, df_trades, self.get_contract_strike)
        except ValueError as ve:
            print(f"Value error when populating trades for : {str(self)}. Were no trades found?")
        
        return df_trades
    
    def __str__(self):
        return str(self.option_selector)

    def __repr__(self):
        return str(self.option_selector)
    
class DualOptionSelectorStaticEntryPolicy(AbstractStaticEntryPolicy):
    def __init__(self, dte, trade_time, bear_option_selector, bull_option_selector, df_trend):
        self.dte = dte
        self.trade_time = trade_time
        self.bear_option_selector = bear_option_selector
        self.bull_option_selector = bull_option_selector
        self.df_trend = df_trend
        super().__init__()
    
    def populateTrades(self, df):
        df_exp = df.loc[df[df['dte']==self.dte].groupby('expiration')['dte'].idxmin()]
        df_dates = df_exp.loc[df_exp.groupby('quote_date')['dte'].idxmin()]
        last_date = df_dates['quote_date'].max()
        df_dates = df_dates[df_dates['expiration']<=last_date]
        df_trades = df_dates[['quote_date','expiration']].copy()
        df_trades.rename({'quote_date': 'trade_date'}, axis='columns', inplace=True)
        df_trades['trade_count'] = 0
        
        bullish_days = self.df_trend[(self.df_trend['x_day_average']>0.00)].reset_index()['quote_date']
        bearish_days = self.df_trend[(self.df_trend['x_day_average']<0.00)].reset_index()['quote_date']
        
        df_trades_bullish = df_trades[df_trades['trade_date'].isin(bullish_days)].copy()
        df_trades_bearish = df_trades[df_trades['trade_date'].isin(bearish_days)].copy()
        
        df_data = df[(df['quote_time'] == self.trade_time) & (df['dte']==self.dte)]
        
        try:
            df_trades_bullish = self.bull_option_selector.populateTrades(df_data, df_trades_bullish, self.get_contract_strike)
            df_trades_bearish = self.bear_option_selector.populateTrades(df_data, df_trades_bearish, self.get_contract_strike)
        except ValueError as ve:
            print(f"Value error when populating trades for : {str(self)}. Were no trades found?")
            
        df_trades = pd.concat([df_trades_bullish,df_trades_bearish])
        df_trades.sort_values(by=['trade_date'], inplace=True)
        
        return df_trades
    
    def __str__(self):
        return str(self.bear_option_selector) + str(self.bull_option_selector)

    def __repr__(self):
        return str(self.bear_option_selector) + str(self.bull_option_selector)
    
# AbstractOptionSelector, DeltaOptionSelector, YieldOptionSelector
class AbstractOptionSelector(ABC):
    @abstractmethod
    def populateTrades(self, df_data, df_trades):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
class DeltaOptionSelector(AbstractOptionSelector):
    def __init__(self, short_puts=None, short_calls=None, long_puts=None, long_calls=None):
        """
        Initializes the Delta Option Selector class. All params are target deltas
        
        :param short_puts: A float or list of deltas to open positions with
        :param short_calls: None/float/list
        :param long_puts: None/float/list
        :param long_calls: None/float/list
        :return: n/a
        """
        # Any of the input variables can be None, float, or a list of floats
        # TODO : Add type checking for list
        self.summary = "Delta "
        
        if short_puts is not None:
            self.short_puts = [short_puts] if type(short_puts) is float else short_puts
            for sp in self.short_puts:
                self.summary += f"SP: {sp:.3f} "
        else:
            self.short_puts = []
            
        if short_calls is not None:
            self.short_calls = [short_calls] if type(short_calls) is float else short_calls
            for sc in self.short_calls:
                self.summary += f"SC: {sc:.3f} "
        else:
            self.short_calls = []
            
        if long_puts is not None:
            self.long_puts = [long_puts] if type(long_puts) is float else long_puts
            for lp in self.long_puts:
                self.summary += f"LP: {lp:.3f} "
        else:
            self.long_puts = []
            
        if long_calls is not None:
            self.long_calls = [long_calls] if type(long_calls) is float else long_calls
            for lc in self.long_calls:
                self.summary += f"LC: {lc:.3f} "
        else:
            self.long_calls = []
            
        

        
    def populateTrades(self, df_data, df_trades, get_contract_strike):
        """
        Populates trades for the classes deltas 0 to n trades per day
        
        :param df_data: must be filtered for the proper date and time of trade entry
        :param df_trades: must contain rows pertaining to the trade dates
        :param get_contract_strike: method for obtaining the strike, data
        :return: df_trades with added columns for 
        """
        if df_trades.shape[0] == 0:
            print("***WARNING*** : No rows were configured in df_trades. No backtest can be generated.")
            
        df_c = df_data[(df_data['type']=="C")].copy()
        df_p = df_data[(df_data['type']=="P")].copy()
        
        if df_c.shape[0] == 0:
            print("***WARNING*** : No calls were found with static rules.")
            
        if df_p.shape[0] == 0:
            print("***WARNING*** : No puts were found with static rules.")
            
        if df_trades.shape[0]==0:
            print("***WARNING*** : No trades are possible during the time frame in the data provided.")
        
            
        # Add all short puts to df_trades
        for i in range(len(self.short_puts)):
            df_trades[f'delta_sp_target_{i}'] = self.short_puts[i]
            df_sp = df_p[df_p['delta']<= self.short_puts[i]]
            df_sp = df_sp.loc[df_sp.groupby(['quote_date'])[['delta']].idxmax()['delta']]
            df_trades[f'strike_sp_{i}'],df_trades[f'delta_sp_{i}'],df_trades[f'collected_sp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_sp_target_{i}'], df_sp,'sell'), axis = 1).T.values
        
        # Add all short calls to df_trades
        for i in range(len(self.short_calls)):
            df_trades[f'delta_sc_target_{i}'] = self.short_calls[i]
            df_sc = df_c[df_c['delta']<= self.short_calls[i]]
            df_sc = df_sc.loc[df_sc.groupby(['quote_date'])[['delta']].idxmax()['delta']]
            df_trades[f'strike_sc_{i}'],df_trades[f'delta_sc_{i}'],df_trades[f'collected_sc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_sc_target_{i}'], df_sc,'sell'), axis = 1).T.values

        # Add all long calls to df_trades
        for i in range(len(self.long_calls)):
            df_trades[f'delta_lc_target_{i}'] = self.long_calls[i]
            df_lc = df_c[df_c['delta']<= self.long_calls[i]]# We don't know for certain how many expirations we have for a given quote date, so lets group on each and find min delta
            df_lc = df_lc.loc[df_lc.groupby(['quote_date'])['delta'].idxmax()]
            df_trades[f'strike_lc_{i}'],df_trades[f'delta_lc_{i}'],df_trades[f'collected_lc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_lc_target_{i}'], df_lc, 'buy'), axis = 1).T.values
            df_trades[f'collected_lc_{i}'] = df_trades[f'collected_lc_{i}']*-1

        # Add all long puts to df_trades
        for i in range(len(self.long_puts)):
            df_trades[f'delta_lp_target_{i}'] = self.long_puts[i]
            df_lp = df_p[df_p['delta']<= self.long_puts[i]]
            df_lp = df_lp.loc[df_lp.groupby(['quote_date'])['delta'].idxmax()]
            df_trades[f'strike_lp_{i}'],df_trades[f'delta_lp_{i}'],df_trades[f'collected_lp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_lp_target_{i}'], df_lp, 'buy'), axis = 1).T.values
            df_trades[f'collected_lp_{i}']=df_trades[f'collected_lp_{i}']*-1

        filt_cols = [col for col in df_trades.columns.to_list() if "strike_" in col]
        df_trades['trade_count'] = df_trades.loc[:,filt_cols].astype(bool).sum(axis=1)
        
        #TODO : Simplify above code because we no longer accept arrays of deltas
        #TODO : Create infeasibility function and refactor this and the other infeasible items from perform calcs into one function
        cols = df_trades.columns
        if 'strike_sp_0' in cols and 'strike_lp_0' in cols:
            df_trades.loc[(df_trades['strike_sp_0'] == df_trades['strike_lp_0'])|(df_trades['collected_sp_0'] <= abs(df_trades['collected_lp_0'])),'trade_count'] -=2
            df_trades.loc[(df_trades['strike_sp_0'] == df_trades['strike_lp_0'])|(df_trades['collected_sp_0'] <= abs(df_trades['collected_lp_0'])),['collected_sp_0','collected_lp_0']] = 0


        if 'strike_sc_0' in cols and 'strike_lc_0' in cols:
            df_trades.loc[(df_trades['strike_sc_0'] == df_trades['strike_lc_0'])|(df_trades['collected_sc_0'] <= abs(df_trades['collected_lc_0'])),'trade_count'] -=2
            df_trades.loc[(df_trades['strike_sc_0'] == df_trades['strike_lc_0'])|(df_trades['collected_sc_0'] <= abs(df_trades['collected_lc_0'])),['collected_sc_0','collected_lc_0']] = 0
        
        filt_cols = [col for col in df_trades.columns.to_list() if "collected_" in col]
        df_trades['collected'] = df_trades.loc[:,filt_cols].sum(axis=1)
            
        return df_trades
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary

class DeltaAndOffsetOptionSelector(AbstractOptionSelector):
    def __init__(self, short_puts=None, short_calls=None, long_put_offsets=None, long_call_offsets=None):
        """
        Initializes the Delta Option Selector class. All params are target deltas
        
        :param short_puts: A float or list of deltas to open positions with
        :param short_calls: None/float/list
        :param long_put_offsets: int/list of ints specifying how many strikes away from the short put
        :param long_call_offsets: int/list of ints specifying how many strikes away from the short call
        :return: n/a
        """
        # Any of the input variables can be None, float, or a list of floats
        # TODO : Add type checking for list
        self.summary = "Delta "
        
        if (short_puts is not None and long_put_offsets is None) or (short_calls is not None and long_call_offsets is None):
            raise ValueError('DeltaAndOffsetOptionSelector requires offsets for each short position.')
        
        
        if short_puts is not None:
            self.short_puts = [short_puts] if type(short_puts) is float else short_puts
            for sp in self.short_puts:
                self.summary += f"SP: {sp:.3f} "
        else:
            self.short_puts = []
            
        if short_calls is not None:
            self.short_calls = [short_calls] if type(short_calls) is float else short_calls
            for sc in self.short_calls:
                self.summary += f"SC: {sc:.3f} "
        else:
            self.short_calls = []
            
        if long_put_offsets is not None:
            self.long_put_offsets = [long_put_offsets] if type(long_put_offsets) is int else long_put_offsets
            for lp in self.long_put_offsets:
                self.summary += f"LP-OS: {lp} "
        else:
            self.long_put_offsets = []
            
        if long_call_offsets is not None:
            self.long_call_offsets = [long_call_offsets] if type(long_call_offsets) is int else long_call_offsets
            for lc in self.long_call_offsets:
                self.summary += f"LC-OS: {lc} "
        else:
            self.long_call_offsets = []
            
        

        
    def populateTrades(self, df_data, df_trades, get_contract_strike):
        """
        Populates trades for the classes deltas 0 to n trades per day
        
        :param df_data: must be filtered for the proper date and time of trade entry
        :param df_trades: must contain rows pertaining to the trade dates
        :param get_contract_strike: method for obtaining the strike, data
        :return: df_trades with added columns for 
        """
        if df_trades.shape[0] == 0:
            print("***WARNING*** : No rows were configured in df_trades. No backtest can be generated.")
            
        df_c = df_data[(df_data['type']=="C")].copy()
        df_p = df_data[(df_data['type']=="P")].copy()
        
        if df_c.shape[0] == 0:
            print("***WARNING*** : No calls were found with static rules.")
            
        if df_p.shape[0] == 0:
            print("***WARNING*** : No puts were found with static rules.")
            
        if df_trades.shape[0]==0:
            print("***WARNING*** : No trades are possible during the time frame in the data provided.")
        
            
        # Add all short puts to df_trades
        for i in range(len(self.short_puts)):
            df_trades[f'delta_sp_target_{i}'] = self.short_puts[i]
            df_sp = df_p[df_p['delta']<= self.short_puts[i]]
            df_sp = df_sp.loc[df_sp.groupby(['quote_date'])[['delta']].idxmax()['delta']]
            df_trades[f'strike_sp_{i}'],df_trades[f'delta_sp_{i}'],df_trades[f'collected_sp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_sp_target_{i}'], df_sp,'sell'), axis = 1).T.values
        
        # Add all short calls to df_trades
        for i in range(len(self.short_calls)):
            df_trades[f'delta_sc_target_{i}'] = self.short_calls[i]
            df_sc = df_c[df_c['delta']<= self.short_calls[i]]
            df_sc = df_sc.loc[df_sc.groupby(['quote_date'])[['delta']].idxmax()['delta']]
            df_trades[f'strike_sc_{i}'],df_trades[f'delta_sc_{i}'],df_trades[f'collected_sc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],row[f'delta_sc_target_{i}'], df_sc,'sell'), axis = 1).T.values

        # Add all long calls to df_trades
        for i in range(len(self.long_call_offsets)):
            df_trades[f'delta_lc_os_{i}'] = self.long_call_offsets[i]
            # repeat the short call filter in order to count offsets
            df_lc = df_c[df_c['delta']<= self.short_calls[i]]
            df_lc = df_lc.loc[df_lc.groupby(['quote_date'])['delta'].nlargest(self.long_call_offsets[i]+1).reset_index().set_index('level_1').groupby('quote_date')['delta'].idxmin()]
            df_trades[f'strike_lc_{i}'],df_trades[f'delta_lc_{i}'],df_trades[f'collected_lc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_lc, 'buy'), axis = 1).T.values
            df_trades[f'collected_lc_{i}'] = df_trades[f'collected_lc_{i}']*-1

        # Add all long puts to df_trades
        for i in range(len(self.long_put_offsets)):
            df_trades[f'delta_lp_os_{i}'] = self.long_put_offsets[i]
            # repeat the short put filter in order to count offsets
            df_lp = df_p[df_p['delta']<= self.short_puts[i]]
            df_lp = df_lp.loc[df_lp.groupby(['quote_date'])['delta'].nlargest(self.long_put_offsets[i]+1).reset_index().set_index('level_1').groupby('quote_date')['delta'].idxmin()]
            df_trades[f'strike_lp_{i}'],df_trades[f'delta_lp_{i}'],df_trades[f'collected_lp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_lp, 'buy'), axis = 1).T.values
            df_trades[f'collected_lp_{i}']=df_trades[f'collected_lp_{i}']*-1

        filt_cols = [col for col in df_trades.columns.to_list() if "strike_" in col]
        df_trades['trade_count'] = df_trades.loc[:,filt_cols].astype(bool).sum(axis=1)
        
        #TODO : Simplify above code because we no longer accept arrays of deltas
        #TODO : Create infeasibility function and refactor this and the other infeasible items from perform calcs into one function
        cols = df_trades.columns
        if 'strike_sp_0' in cols and 'strike_lp_0' in cols:
            df_trades.loc[(df_trades['strike_sp_0'] == df_trades['strike_lp_0'])|(df_trades['collected_sp_0'] <= abs(df_trades['collected_lp_0'])),'trade_count'] -=2
            df_trades.loc[(df_trades['strike_sp_0'] == df_trades['strike_lp_0'])|(df_trades['collected_sp_0'] <= abs(df_trades['collected_lp_0'])),['collected_sp_0','collected_lp_0']] = 0


        if 'strike_sc_0' in cols and 'strike_lc_0' in cols:
            df_trades.loc[(df_trades['strike_sc_0'] == df_trades['strike_lc_0'])|(df_trades['collected_sc_0'] <= abs(df_trades['collected_lc_0'])),'trade_count'] -=2
            df_trades.loc[(df_trades['strike_sc_0'] == df_trades['strike_lc_0'])|(df_trades['collected_sc_0'] <= abs(df_trades['collected_lc_0'])),['collected_sc_0','collected_lc_0']] = 0
        
        filt_cols = [col for col in df_trades.columns.to_list() if "collected_" in col]
        df_trades['collected'] = df_trades.loc[:,filt_cols].sum(axis=1)
            
        return df_trades
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
class YieldOptionSelector(AbstractOptionSelector):
    def __init__(self, short_puts=None, short_calls=None, long_puts=None, long_calls=None, ipv=None):
        """
        Initializes the Delta Option Selector class. All params are target deltas
        
        :param short_puts: A float or list of deltas to open positions with
        :param short_calls: None/float/list
        :param long_puts: None/float/list
        :param long_calls: None/float/list
        :return: n/a
        """
        # Any of the input variables can be None, float, or a list of floats
        # TODO : Add type checking for list
        
        if ipv is None:
            raise TypeError("Argument ipv cannot be NoneType. A non-zero value must be specified.")
        self.ipv = ipv
        
        self.summary = "Yields "
        self.target_tol = 1.2
        
        if short_puts is not None:
            self.short_puts = [short_puts] if type(short_puts) is float else short_puts
            for sp in self.short_puts:
                self.summary += f"SP: {sp:.3%} "
        else:
            self.short_puts = []
            
        if short_calls is not None:
            self.short_calls = [short_calls] if type(short_calls) is float else short_calls
            for sc in self.short_calls:
                self.summary += f"SC: {sc:.3%} "
        else:
            self.short_calls = []
            
        if long_puts is not None:
            self.long_puts = [long_puts] if type(long_puts) is float else long_puts
            for lp in self.long_puts:
                self.summary += f"LP: {lp:.3%} "
        else:
            self.long_puts = []
            
        if long_calls is not None:
            self.long_calls = [long_calls] if type(long_calls) is float else long_calls
            for lc in self.long_calls:
                self.summary += f"LC: {lc:.3%} "
        else:
            self.long_calls = []

        
    def populateTrades(self, df_data, df_trades, get_contract_strike):
        """
        Populates trades for the classes deltas 0 to n trades per day
        
        :param df_data: must be filtered for the proper date and time of trade entry
        :param df_trades: must contain rows pertaining to the trade dates
        :param get_contract_strike: method for obtaining the strike, data
        :return: df_trades with added columns for 
        """
        if df_trades.shape[0] == 0:
            print("***WARNING*** : No rows were configured in df_trades. No backtest can be generated.")
            
        df_c = df_data[(df_data['type']=="C")].copy()
        df_p = df_data[(df_data['type']=="P")].copy()
        
        if df_c.shape[0] == 0:
            print(f"***WARNING*** : No calls were found with static rules.")
            
        if df_p.shape[0] == 0:
            print(f"***WARNING*** : No puts were found with static rules.")
        
            
        # Add all short puts to df_trades
        for i in range(len(self.short_puts)):
            target = self.ipv*self.short_puts[i] / 100
            df_trades[f'yield_sp_target_{i}'] = self.short_puts[i]
            df_sp = df_p[df_p['ask']+df_p['bid'] < target*2*self.target_tol]
            df_sp = df_sp.loc[df_sp.groupby(['quote_date'])['strike'].idxmax()]
            df_trades[f'strike_sp_{i}'],df_trades[f'delta_sp_{i}'],df_trades[f'collected_sp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_sp,'sell'), axis = 1).T.values
        
        # Add all short calls to df_trades
        for i in range(len(self.short_calls)):
            target = self.ipv*self.short_calls[i] / 100
            df_trades[f'yield_sc_target_{i}'] = self.short_calls[i]
            df_sc = df_c[df_c['ask'] + df_c['bid']< target*2*self.target_tol]
            df_sc = df_sc.loc[df_sc.groupby(['quote_date'])['strike'].idxmin()]            
            df_trades[f'strike_sc_{i}'],df_trades[f'delta_sc_{i}'],df_trades[f'collected_sc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_sc,'sell'), axis = 1).T.values

        # Add all long calls to df_trades
        for i in range(len(self.long_calls)):
            target = self.ipv*self.long_calls[i] / 100
            df_trades[f'yield_lc_target_{i}'] = self.long_calls[i]
            df_lc = df_c[df_c['ask'] + df_c['bid']< target*2*self.target_tol]# We don't know for certain how many expirations we have for a given quote date, so lets group on each and find min delta
            df_lc = df_lc.loc[df_lc.groupby(['quote_date'])['strike'].idxmin()]
            df_trades[f'strike_lc_{i}'],df_trades[f'delta_lc_{i}'],df_trades[f'collected_lc_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_lc, 'buy'), axis = 1).T.values
            df_trades[f'collected_lc_{i}'] = df_trades[f'collected_lc_{i}']*-1

        # Add all long puts to df_trades
        for i in range(len(self.long_puts)):
            df_trades[f'yield_lp_target_{i}'] = self.long_puts[i]
            df_lp = df_p[df_p['ask']+df_p['bid'] < target*2*self.target_tol]
            df_lp = df_lp.loc[df_lp.groupby(['quote_date'])['strike'].idxmax()]
            df_trades[f'strike_lp_{i}'],df_trades[f'delta_lp_{i}'],df_trades[f'collected_lp_{i}'] = df_trades.apply(
                lambda row : get_contract_strike(row['trade_date'], row['expiration'],None, df_lp, 'buy'), axis = 1).T.values
            df_trades[f'collected_lp_{i}']=df_trades[f'collected_lp_{i}']*-1

        filt_cols = [col for col in df_trades.columns.to_list() if "strike_" in col]
        df_trades['trade_count'] = df_trades.loc[:,filt_cols].astype(bool).sum(axis=1)
        
        filt_cols = [col for col in df_trades.columns.to_list() if "collected_" in col]
        df_trades['collected'] = df_trades.loc[:,filt_cols].sum(axis=1)
            
        return df_trades
    
    
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
#####
# Dynamic Entry Policies
#####


class OptionSelectorDynamicEntryPolicy(AbstractDynamicEntryPolicy):
    def __init__(self, dte, option_selector, dteMatch = 'eq'):
        self.dte = dte
        self.option_selector = option_selector
        
        dteMatchValues = ['leq','eq','geq']
        if dteMatch not in dteMatchValues:
            raise ValueError(f"Invalid argument passed in for dteMatch : {dteMatch}. Value must be one of {dteMatchValues}")
            
        self.dteMatch = dteMatch
        super().__init__()
    
    def evaluateTradingCondition(self, eval_date, eval_time, df_trade_row, df_data):
        df_trades = None
        if (df_trade_row is None or (df_trade_row['expiration'] == eval_date and eval_time=='16:00:00') or df_trade_row['expiration'] < eval_date or df_trade_row['lost_p']>0 or df_trade_row['lost_c']>0):
            #populate a new trade
            trade_count = 0
            expiration_p, expiration_c = None, None

            # create call/put dataframe slice if needed
            if self.dteMatch == 'geq':
                df_purchase_time = df_data[(df_data['dte']>=self.dte) & (df_data['quote_time'] == eval_time)]
            elif self.dteMatch == 'eq':
                df_purchase_time = df_data[(df_data['dte']==self.dte) & (df_data['quote_time'] == eval_time)]
            else: 
                # The last option is dteMatch =='leq', meaning we will allow trades that are 0 dte up to (and including) the target dte
                df_purchase_time = df_data[(df_data['dte']<=self.dte) & (df_data['quote_time'] == eval_time)]
                # If the evaluation time is in the last 15 minutes, we don't want 0-DTE
                if eval_time>='15:45:00':
                    df_purchase_time = df_purchase_time[df_purchase_time['dte']>0]
                
            exp = df_purchase_time['expiration'].min()
            df_purchase_time = df_purchase_time[df_purchase_time['expiration']==exp]
            
            if df_purchase_time.shape[0] == 0:
                return None
                    
            df_trades = pd.DataFrame([{'trade_date':eval_date, 'expiration':exp, 'trade_count':trade_count, 'lost_c':0.0, 'lost_p':0.0}])
        
            return self.option_selector.populateTrades(df_purchase_time, df_trades, self.get_contract_strike)
        
#         print(f"Pre-conditions not satisfied. Eval date {eval_date}. eval time : {eval_time}")
#         print(f"df trade row type : {type(df_trade_row)}")
#         print(f"df trade row : {df_trade_row}")
#         print(f"eval date {eval_date}")
#         print(f"eval time {eval_time}")
              
        return None
    
    def __str__(self):
        return f"{str(self.option_selector)} DTE: {self.dte} Match: {self.dteMatch}"

    def __repr__(self):
        return f"{str(self.option_selector)} DTE: {self.dte} Match: {self.dteMatch}"
    
    
#### Basic Strategy Types

class StaticEntryDynamicExitStrategy(AbstractStrategy):
    def __init__(self, commission, df_ty, ipv, entry_policy, exit_policy, debug = False):
        if not isinstance(entry_policy, AbstractStaticEntryPolicy):
            raise TypeError("Argument 'entry_policy' must be a subclass of 'AbstractStaticEntryPolicy'.")
            
        if not isinstance(exit_policy, AbstractDynamicExitPolicy):
            raise TypeError("Argument 'exit_policy' must be a subclass of 'AbstractDynamicExitPolicy'.")
            
        self.summary = f"{str(entry_policy)} {str(exit_policy)}"
        super().__init__(entry_policy, exit_policy, df_ty, ipv, commission, debug)

    def evaluate(self,df):
        self.df_trades = self.entry_policy.populateTrades(df)
        self.df = df
        
        self.df_trades['lost_c'] = 0.0
        self.df_trades['lost_p'] = 0.0
        
        df_times = pd.DataFrame(df['quote_time'].unique())
        df_times[0] = pd.to_datetime(df_times[0], format='%H:%M:%S').dt.time
        df_times.rename(columns={0: "quote_time"},inplace=True)
        df_times = df_times.sort_values(by='quote_time')
        
        start_time_timer = perf_counter()
        for index, trow in df_times.iterrows():           
            min_i = int(trow[0].strftime("%M"))
            curr_time = trow[0]
            start_time = time(hour=9, minute=30)
            end_time = time(hour=16)
            if (end_time < curr_time or curr_time <= start_time):
                continue
            str_quote_time = trow[0].strftime("%H:%M:%S")
            df_qt = df[(df['quote_time']==str_quote_time)]

            # print(f"Num rows at {str_quote_time} : {df_qt.shape[0]}")

            self.df_trades['trade_count'],self.df_trades['lost_c'], self.df_trades['lost_p'] = self.df_trades.apply(
                lambda row : self.exit_policy.evaluateTradingCondition(None, str_quote_time, row, df_qt), axis=1).T.values
            stop_time_timer = perf_counter()
            # print(df_trades.head())

            if min_i % 30 == 0:
                print(f"Processed {curr_time} | Elapsed : {stop_time_timer - start_time_timer}")
                start_time_timer = stop_time_timer
        
        return self.performCalcs()


    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
class DynamicEntryDynamicExitStrategy(AbstractStrategy):
    def __init__(self, commission, df_ty, ipv, entry_policy, exit_policy, debug = False):
        if not isinstance(entry_policy, AbstractDynamicEntryPolicy):
            raise TypeError("Argument 'entry_policy' must be a subclass of 'AbstractDynamicEntryPolicy'.")
        self.entry_policy = entry_policy
            
        if not isinstance(exit_policy, AbstractDynamicExitPolicy):
            raise TypeError("Argument 'exit_policy' must be a subclass of 'AbstractDynamicExitPolicy'.")
        self.exit_policy = exit_policy
        
        self.summary = f"{str(entry_policy)} {str(exit_policy)}"
        
        self.df_trades = pd.DataFrame(columns=['trade_date', 'expiration', 'trade_count', 'collected', 'lost_c', 'lost_p'])
        super().__init__(entry_policy, exit_policy, df_ty, ipv, commission, debug)

    def evaluate(self,df):
        self.df = df
        df.loc[:,'quote_datetime'] = pd.to_datetime(df['quote_date'].astype(str)+' '+df['quote_time'].astype(str))
        df_datetimes = pd.DataFrame(df['quote_datetime'].unique())
        df_datetimes.rename(columns={0: "quote_datetime"},inplace=True)
        df_datetimes = df_datetimes.sort_values(by='quote_datetime')

        start_time_timer = perf_counter()
        df_dt = None
        for index, trow in df_datetimes.iterrows():
            day_i = int(trow[0].strftime("%d"))
            hour_i = int(trow[0].strftime("%H"))

            start_time = time(hour=9, minute=30)
            end_time = time(hour=16)
            curr_time = trow[0].time()
            str_quote_time = curr_time.strftime("%H:%M:%S")
            curr_date = pd.to_datetime(trow[0]).floor('D')

            if (end_time < curr_time or curr_time < start_time):
                continue

            # To save computation, pre-filter each time the date changes
            if df_dt is None or df_dt['quote_date'].iloc[0] != curr_date:
                df_dt = df[df['quote_date']==curr_date]
                
            # print(f"Iteration date : {curr_date} and time {str_quote_time}. DF rows {df.shape[0]}. df_dt rows : {df_dt.shape[0]}")

            # Design decision : we'll just try to enter and exit once each minute. This means that if we exit a position,
            # about 1 minute will pass before we decide to re-enter
            # 
            # To keep evalute trading decision method signature the same, we'll pass in a row: the last row of df_trades.
            # This leaves the possibility of having no rows.
            # Thus, every dynamic policy must account for getting no row.
            last_trade = None
            if self.df_trades.shape[0] > 0:
                last_trade = self.df_trades.iloc[-1]
            new_trade = self.entry_policy.evaluateTradingCondition(curr_date, str_quote_time, last_trade, df_dt)
            if new_trade is not None:
                self.df_trades = pd.concat([self.df_trades, new_trade], ignore_index=True)

            if self.df_trades.shape[0] > 0:
                last_trade = self.df_trades.iloc[-1]
            if last_trade is not None:
                if last_trade['expiration'] >= curr_date:
                    #print(f"call : evaluateTradingCondition {curr_date}, {str_quote_time}, {last_trade['expiration']}, {df_dt.shape[0]}")
                    trade_count, lost_c, lost_p = self.exit_policy.evaluateTradingCondition(curr_date, str_quote_time, last_trade, df_dt)
#                     if lost_c > 0 or lost_p > 0:
#                         print("Trying to propagate a loss into the future.")
                    self.df_trades.loc[self.df_trades.index[-1],'trade_count'] = trade_count
                    self.df_trades.loc[self.df_trades.index[-1],'lost_c'] = lost_c
                    self.df_trades.loc[self.df_trades.index[-1],'lost_p'] = lost_p

            
            stop_time_timer = perf_counter()

            if day_i == 28 and hour_i == 16:
                print(f"Processed {trow.iloc[0]} | Elapsed : {stop_time_timer - start_time_timer}")
                start_time_timer = stop_time_timer
        
        return self.performCalcs()


    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
class StaticEntryStaticExitStrategy(AbstractStrategy):
    def __init__(self, commission, df_ty, ipv, entry_policy, exit_policy, debug=False):
        if not isinstance(entry_policy, AbstractStaticEntryPolicy):
            raise TypeError("Argument 'entry_policy' must be a subclass of 'AbstractStaticEntryPolicy'.")
            
        if not isinstance(exit_policy, AbstractStaticExitPolicy):
            raise TypeError("Argument 'exit_policy' must be a subclass of 'AbstractStaticExitPolicy'.")
        self.summary = f"{str(entry_policy)} {str(exit_policy)}"
        super().__init__(entry_policy, exit_policy, df_ty, ipv, commission, debug)
        
    def evaluate(self,df):
        self.df = df
        self.df_trades = self.entry_policy.populateTrades(df)
        self.df_trades = self.exit_policy.populateTrades(df, self.df_trades)
        return self.performCalcs()
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
    
class AbstractExitPolicy(AbstractPolicy):   
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
    
class AbstractDynamicExitPolicy(AbstractExitPolicy):
    @abstractmethod
    def populateTrades(self, df):
        pass
    
    @abstractmethod
    def evaluateTradingCondition(self, eval_date, eval_time, df_trade_row, df_data):
        pass
    
    def __str__(self):
        pass

    def __repr__(self):
        pass
    

    
class AbstractStaticExitPolicy(AbstractExitPolicy):
    def __init__(self, trade_time):
        self.trade_time = trade_time
        super().__init__()
    
    @abstractmethod
    def populateTrades(self, df):
        pass
    
    def __str__(self):
        pass

    def __repr__(self):
        pass
    

class HoldToExpStaticExitPolicy(AbstractStaticExitPolicy):
    def __init__(self):
        self.summary = f"Hold Until Expiration"
        super().__init__('16:00:00')
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
    def populateTrades(self, df, df_trades):
        df_qt = df[(df['quote_time']==self.trade_time)&(df['dte']==0)]
        df_trades['lost_c'], df_trades['lost_p'] = df_trades.apply(
            lambda row : self.get_amount_lost(df_qt, row), axis=1).T.values
        return df_trades
    
class ExitAtTimeStaticExitPolicy(AbstractStaticExitPolicy):
    def __init__(self, trade_time):
        self.summary = f"Exit position at {trade_time}"
        super().__init__(trade_time)
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
    def evaluateTradingCondition(self, df_trade_row, df_data):
        row = df_trade_row
        expiration, trade_count  = row['expiration'], row['trade_count']
        
        if self.trade_time == '16:00:00':
            lost_c, lost_p = self.get_amount_lost(df_data, df_trade_row)
            return pd.Series([trade_count,lost_c,lost_p])
        
        eval_time = df_data['quote_time'].iloc[0]

        mark_to_market_c = 0.0
        mark_to_market_p = 0.0
        df1 = df_data[(df_data['quote_date'] == expiration) & (df_data['expiration'] == expiration)]
        df_c = df1[df1['type']=='C']
        df_p = df1[df1['type']=='P']
        potential_trade_count = 0
        for col in row.index.values:
            opt_type = 'C'
            if 'strike_sc' in col and row[col] != 0.0:
                    df_sc=df_c[df_c['strike']==row[col]]
                    self.len_check(df_sc, 'C', row[col], expiration, eval_time)
                    s_sc = df_sc.iloc[0]
                    mark_to_market_c -= self.getRoundedSlippedPrice(s_sc['ask'],s_sc['bid'],'buy')
                    potential_trade_count +=1
            if 'strike_lc' in col and row[col] != 0.0:
                    df_lc=df_c[df_c['strike']==row[col]]
                    self.len_check(df_lc, 'C', row[col], expiration, eval_time)
                    s_lc = df_lc.iloc[0]
                    mark_to_market_c += self.getRoundedSlippedPrice(s_lc['ask'],s_lc['bid'],'sell')
                    potential_trade_count +=1
            opt_type='P'
            if 'strike_sp' in col and row[col] != 0.0:
                    df_sp=df_p[df_p['strike']==row[col]]
                    self.len_check(df_sp, 'P', row[col], expiration, eval_time)
                    s_sp = df_sp.iloc[0]
                    mark_to_market_p -= self.getRoundedSlippedPrice(s_sp['ask'],s_sp['bid'],'buy')
                    potential_trade_count +=1
            if 'strike_lp' in col and row[col] != 0.0:
                    df_lp=df_p[df_p['strike']==row[col]]
                    self.len_check(df_lp, 'P', row[col], expiration, eval_time)
                    s_lp = df_lp.iloc[0]
                    mark_to_market_p += self.getRoundedSlippedPrice(s_lp['ask'],s_lp['bid'],'sell')
                    potential_trade_count +=1
                    
        trade_count += potential_trade_count
                
        return pd.Series([trade_count,-1*mark_to_market_c,-1*mark_to_market_p])
    
    def populateTrades(self, df, df_trades):
        df_qt = df[(df['quote_time']==self.trade_time)&(df['dte']==0)]
        df_trades['trade_count'],df_trades['lost_c'], df_trades['lost_p'] = df_trades.apply(
            lambda row : self.evaluateTradingCondition(row, df_qt), axis=1).T.values
        return df_trades
    
    
    
class MaxLossDynamicExitPolicy(AbstractDynamicExitPolicy):
    def __init__(self, max_loss):
        self.max_loss = max_loss
        self.drop_both_legs = False
        super().__init__()
    
    def populateTrades(self, df):
        pass
    
    def evaluateTradingCondition(self, eval_date, eval_time, df_trade_row, df_data):
        row = df_trade_row
        expiration, collected, lost_c, lost_p, trade_count  = row['expiration'], row['collected'], row['lost_c'], row['lost_p'], row['trade_count']
        if lost_c > 0.0 or lost_p > 0.0:
            return pd.Series([trade_count, lost_c, lost_p])
        
        if (eval_time == '16:00:00' and eval_date == row['expiration']):
            #print(f"evaluating end of day")
            lost_c, lost_p = self.get_amount_lost(df_data,row)
#             if lost_c>0 or lost_p>0:
#                 print(f"We have a loss! eval_date {eval_date}, {df_trade_row}")
            return pd.Series([trade_count,lost_c, lost_p])

        eval_date = eval_date or expiration
        
        df1 = df_data[(df_data['quote_date'] == eval_date) & (df_data['expiration'] == expiration) & (df_data['quote_time']==eval_time)]
#         print(f"Num rows at {expiration} : {df1.shape[0]}")
        mark_to_market_c = 0.0
        mark_to_market_p = 0.0
        df_c = df1[df1['type']=='C']
        df_p = df1[df1['type']=='P']
        potential_trade_count = 0
        for col in row.index.values:
            opt_type = 'C'
            if 'strike_sc' in col and row[col] != 0.0:
                    df_sc=df_c[df_c['strike']==row[col]]
                    self.len_check(df_sc, 'C', row[col], expiration, eval_time)
                    s_sc = df_sc.iloc[0]
                    mark_to_market_c -= self.getRoundedSlippedPrice(s_sc['ask'],s_sc['bid'],'buy')
                    potential_trade_count +=1
            if 'strike_lc' in col and row[col] != 0.0:
                    df_lc=df_c[df_c['strike']==row[col]]
                    self.len_check(df_lc, 'C', row[col], expiration, eval_time)
                    s_lc = df_lc.iloc[0]
                    mark_to_market_c += self.getRoundedSlippedPrice(s_lc['ask'],s_lc['bid'],'sell')
                    potential_trade_count +=1
            opt_type='P'
            if 'strike_sp' in col and row[col] != 0.0:
                    df_sp=df_p[df_p['strike']==row[col]]
                    self.len_check(df_sp, 'P', row[col], expiration, eval_time)
                    s_sp = df_sp.iloc[0]
                    mark_to_market_p -= self.getRoundedSlippedPrice(s_sp['ask'],s_sp['bid'],'buy')
                    potential_trade_count +=1
            if 'strike_lp' in col and row[col] != 0.0:
                    df_lp=df_p[df_p['strike']==row[col]]
                    self.len_check(df_lp, 'P', row[col], expiration, eval_time)
                    s_lp = df_lp.iloc[0]
                    mark_to_market_p += self.getRoundedSlippedPrice(s_lp['ask'],s_lp['bid'],'sell')
                    potential_trade_count +=1
        
        potential_loss = collected + mark_to_market_c + mark_to_market_p
        
        #print(f"Potential Loss : {potential_loss} | collected : {collected}")
        
        if (potential_loss < 0) and (abs(potential_loss/collected) >= self.max_loss):
#             print(f"Exp: {expiration}. Eval time : {eval_time}. Collected : {collected}, MTMC : {mark_to_market_c}, MTMP : {mark_to_market_p}. Projected loss : {abs(potential_loss/collected)}")
#             print(df_sp.head(5))
            trade_count += potential_trade_count
            lost_c += -1*mark_to_market_c
            lost_p += -1*mark_to_market_p
#             print(f"early exit loss! : {row}, eval_date : {eval_date}. eval_time : {eval_time}")
                
        return pd.Series([trade_count,lost_c,lost_p])
    
    
    def __str__(self):
        return f"Max Loss: {self.max_loss}"

    def __repr__(self):
        return f"Max Loss: {self.max_loss}"
    
