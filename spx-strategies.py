# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2022-12-20T16:30:10.623097Z","iopub.execute_input":"2022-12-20T16:30:10.623508Z","iopub.status.idle":"2022-12-20T16:30:10.653377Z","shell.execute_reply.started":"2022-12-20T16:30:10.623476Z","shell.execute_reply":"2022-12-20T16:30:10.652090Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from abc import ABC, abstractmethod

class Strategy:
    def __init__(self, trade_dates):
        self.df = pd.DataFrame(trade_dates)
        
    def get_contract_strike(self, curr_date, contract_date, target_delta, i_df):
        # -1 means no delta should be selected
        if target_delta ==-1:
            return pd.Series([0,0,0])
#         df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date)].min()
        df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date)]
        price = (df1['ask']+df1['bid'])/0.02 # 0.02 is dividing by two and multiplying by 100
        if np.isnan(price):
            return pd.Series([0,0,0])
        rounded_price = 5 * round(price / 5)
        return pd.Series([df1['strike'],df1['delta'],rounded_price])

    def get_collected_from_strike(self, curr_date, contract_date, strike, i_df):   
#         df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date) & (i_df['strike']==strike)].min()
        df1 = i_df[(i_df['quote_date']==curr_date) & (i_df['expiration']==contract_date) & (i_df['strike']==strike)]
        price = (df1['ask']+df1['bid'])/0.02 # 0.02 is dividing by two and multiplying by 100
        if np.isnan(price):
            return pd.Series([0,0,0])
        rounded_price = 5 * round(price / 5)
        return pd.Series([df1['strike'],df1['delta'],rounded_price])
        
    @abstractmethod
    def get_deltas(self):
        pass
    
    def populate_trades(self, df, df_trades):
        purchase_time = '09:45:00'
        df_var_delta = self.get_deltas()
        df = pd.merge(df, df_var_delta, left_on='expiration', right_index=True)
        df_trades = pd.merge(df_trades, df_var_delta, left_on='expiration', right_index=True)
        
        # create call/put dataframe slice if needed
        df_purchase_time = df[(df['quote_time'] == purchase_time)]
        df_c = df_purchase_time[(df_purchase_time['type']=="C")]
        df_p = df_purchase_time[(df_purchase_time['type']=="P")]
        
        ##### SHORT CALL #####
        df_sc = df_c[(df_c['delta']<= df_c['delta_sc_target'])]
        df_sc = df_sc.loc[df_sc.groupby(['quote_date'])[['delta']].idxmax()['delta']]
        df_trades['strike_sc'],df_trades['delta_sc'],df_trades['collected_sc'] = df_trades.apply(
            lambda row : self.get_contract_strike(row['trade_date'], row['expiration'],row['delta_sc_target'], df_sc), axis = 1).T.values
        
        #### SHORT PUT ####
        df_sp = df_p[(df_p['delta']<= df_p['delta_sp_target'])]
        df_sp = df_sp.loc[df_sp.groupby(['quote_date'])[['delta']].idxmax()['delta']]
        df_trades['strike_sp'],df_trades['delta_sp'],df_trades['collected_sp'] = df_trades.apply(
            lambda row : self.get_contract_strike(row['trade_date'], row['expiration'],row['delta_sp_target'], df_sp), axis = 1).T.values
        
        ###### LONG CALL #####
        if (df_trades['lc_offset'] > 0).all():
            df_trades['strike_lc'] = df_trades['strike_sc'] + df_trades['lc_offset']
            df_trades['strike_lc'],df_trades['delta_lc'],df_trades['collected_lc'] = df_trades.apply(
                lambda row : self.get_collected_from_strike(row['trade_date'], row['expiration'],row['strike_lc'], df_c), axis = 1).T.values
            df_trades['collected_lc'] = df_trades['collected_lc']*-1
        else:
            df_lc = df_c[(df_c['delta']>= df_c['delta_lc_target'])]# We don't know for certain how many expirations we have for a given quote date, so lets group on each and find min delta
            df_lc = df_lc.loc[df_lc.groupby(['quote_date'])[['delta']].idxmin()['delta']]
            df_lc.to_csv('df_lp_test.csv')
            df_trades['strike_lc'],df_trades['delta_lc'],df_trades['collected_lc'] = df_trades.apply(
                lambda row : self.get_contract_strike(row['trade_date'], row['expiration'],row['delta_lc_target'], df_lc), axis = 1).T.values
            df_trades['collected_lc'] = df_trades['collected_lc']*-1
            
        ###### LONG PUT #####
        if (df_trades['lp_offset'] > 0).all():
            df_trades['strike_lp'] = df_trades['strike_sp'] - df_trades['lp_offset']
            df_trades['strike_lp'],df_trades['delta_lp'],df_trades['collected_lp'] = df_trades.apply(
                lambda row : self.get_collected_from_strike(row['trade_date'], row['expiration'],row['strike_lp'], df_p), axis = 1).T.values
            df_trades['collected_lp']=df_trades['collected_lp']*-1
        else:
            df_lp = df_p[(df_p['delta']>= df_p['delta_lp_target'])]
            df_lp = df_lp.loc[df_lp.groupby(['quote_date'])[['delta']].idxmin()['delta']]
            df_trades['strike_lp'],df_trades['delta_lp'],df_trades['collected_lp'] = df_trades.apply(
                lambda row : self.get_contract_strike(row['trade_date'], row['expiration'],row['delta_lp_target'], df_lp), axis = 1).T.values
            df_trades['collected_lp']=df_trades['collected_lp']*-1
            
        #TODO - clean this up. 
        # When selling options, short puts will never have lower strikes than long puts. 
        # Likewise with short calls being higher than long calls. 
        # For the time being we will iterate and fix all short positions, but we need a better solution for capability of going long.
        for index, row in df_trades.iterrows():
            if row['strike_lp'] >= row['strike_sp'] and row['strike_sp'] > 0.0:
                df_trades.loc[index,['collected_lp','collected_sp','strike_lp','strike_sp']] = 0.0
            if row['strike_sc'] >= row['strike_lc'] and row['strike_sc'] > 0.0:
                df_trades.loc[index,['collected_lc','collected_sc','strike_lc','strike_sc']] = 0.0
            if df_trades.loc[index,'strike_lc'] > 0.0:
                df_trades.loc[index,'trade_count']+=1
            if df_trades.loc[index,'strike_sc'] > 0.0:
                df_trades.loc[index,'trade_count']+=1
            if df_trades.loc[index,'strike_lp'] > 0.0:
                df_trades.loc[index,'trade_count']+=1
            if df_trades.loc[index,'strike_sp'] > 0.0:
                df_trades.loc[index,'trade_count']+=1
                
        df_trades['collected'] = df_trades['collected_sc'] + df_trades['collected_lc'] + df_trades['collected_sp'] + df_trades['collected_lp']
        return df_trades
        
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    
class StaticStrategyTest(Strategy):
    def __init__(self, trade_dates):
        self.summary = "Static Strategy : "
        super().__init__(trade_dates)
    
    def get_deltas(self):
        self.df.rename({0: "expiration"}, axis='columns', inplace=True)
        self.df.set_index('expiration',inplace=True)
        self.df['delta_sp_target'] = 0.22
        self.df['delta_lp_target'] = None
        self.df['delta_sc_target'] = 0.02
        self.df['delta_lc_target'] = 0.005
        self.df['lp_offset'] = 20.0
        self.df['lc_offset'] = 0
        self.df.fillna(-1,inplace=True)
        return self.df
    
    def populate_trades(self):
        return None
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary

    
class SpxThetaStrategy(Strategy):
    def __init__(self, trade_dates, params):
        self.w_spx = params['w_spx']
        self.w_vix = params['w_vix']
        self.newline = '\n'
        self.summary = f"spxtheta.com Strategy {self.newline} spx w : {self.w_spx}; spx w : {self.w_vix}"
        super().__init__(trade_dates)
    
    def get_deltas(self):
        self.df.rename({0: "expiration"}, axis='columns', inplace=True)
        self.df.set_index('expiration',inplace=True)
        self.df['delta_sp_target'] = None
        self.df['delta_lp_target'] = None
        self.df['delta_sc_target'] = None
        self.df['delta_lc_target'] = None
        self.df['lp_offset'] = 0
        self.df['lc_offset'] = 0
        self.df.fillna(-1,inplace=True)
        return self.df
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
    
class OptimalIronCondorStrategy(Strategy):
    def __init__(self, trade_dates, params):
#         if not params.has_key('delta_sp') or not params.has_key('delta_lp') or not params.has_key('delta_sc') or not params.has_key('delta_lc'):
#             raise Exception("params must contain the following keys : delta_sp, delta_lp, delta_sc, delta_lc")
        self.delta_sp = params['delta_sp']
        self.delta_lp = params['delta_sp'] - params['delta_p_offset'] if params['delta_sp'] - params['delta_p_offset'] > 0.0 else 0
        self.delta_sc = params['delta_sc']
        self.delta_lc = params['delta_sc'] - params['delta_c_offset'] if params['delta_sc'] - params['delta_c_offset'] > 0.0 else 0
        self.summary = f"Optimal Iron Condor Strategy | DSP : {self.delta_sp}; DLP : {self.delta_lp}; DSC : {self.delta_sc}; DLC : {self.delta_lc}"
        super().__init__(trade_dates)
    
    def get_deltas(self):
        self.df.rename({0: "expiration"}, axis='columns', inplace=True)
        self.df.set_index('expiration',inplace=True)
        self.df['delta_sp_target'] = self.delta_sp
        self.df['delta_lp_target'] = self.delta_lp
        self.df['delta_sc_target'] = self.delta_sc
        self.df['delta_lc_target'] = self.delta_lc
        self.df['lp_offset'] = 0
        self.df['lc_offset'] = 0
        self.df.fillna(-1,inplace=True)
        return self.df
    
    def __str__(self):
        return self.summary

    def __repr__(self):
        return self.summary
