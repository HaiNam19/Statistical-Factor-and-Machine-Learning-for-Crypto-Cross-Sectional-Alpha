import pandas as pd
import numpy as np


class feature_engineering:
    def __init__(self, df):
        self.df = df
        
    def build_momentum(self, windows = [7,14,30,90]):
        table = self.df.copy()
            
        table["log_return"] = np.log(table["close"]) - np.log(table.groupby("symbol")["close"].shift(1)) #daily log return
        for window in windows:
            table[f"momentum_{window}d"] = table.groupby("symbol")["log_return"].transform(
                lambda x: x.rolling(window= window, min_periods = window).sum())
            
        temp = table[["timestamp", "symbol", "momentum_7d", "momentum_14d", "momentum_30d", "momentum_90d"]]
        return pd.DataFrame(temp)
    
    def build_reversal(self):
        table = self.df.copy()
        table["reversal_1d"] = np.log(table["close"]) - np.log(table.groupby("symbol")["close"].shift(1))
        table["reversal_3d"] = np.log(table["close"]) - np.log(table.groupby("symbol")["close"].shift(3))
        return table[["timestamp","symbol","reversal_1d","reversal_3d"]]

    def build_vol(self, windows = [7,14,30]):
        table = self.df.copy()
        table["log_return"] = np.log(table["close"]) - np.log(table.groupby("symbol")["close"].shift(1)) 
            
        for window in windows:
            table[f"vol_{window}d"] = table.groupby("symbol")["log_return"].transform(lambda x: x.rolling
                                                                                          (window= window, min_periods = window).std())
                
        table["vol_of_vol_14d"] = table.groupby("symbol")["vol_14d"].transform(lambda x: x.rolling(window = 14,
                                                                                                      min_periods = 14).std())    
            
        return table[["timestamp","symbol","vol_7d","vol_14d","vol_30d","vol_of_vol_14d"]]
    
    def build_liquid(self, windows [14,30]):
        table = df.copy()
        table["log_return"] = np.log(table["close"]) - np.log(table.groupby("symbol")["close"].shift(1))
            
        table["amihud_daily"] = abs(table["log_return"]) / table["dollar_volume"]
            
        for window in windows:
            table[f"amihud_{window}d"] = table.groupby("symbol")["amihud_daily"].transform(lambda x: x.rolling(window = window,
                                                                                                                   min_periods = window).mean())
        
        return table[["timestamp","symbol","log_return","amihud_14d", "amihud_30d"]]
    
    
    def build_factor(self):
        momentum = self.build_momentum(self.df)
        reversal = self.build_reversal(self.df)
        vol = self.build_vol(self.df)
        liquid = self.build_liquid(self.df)
            
        table = self.df.merge(momentum, on = ["timestamp","symbol"], how = 'inner').merge(reversal,
                                    on = ["timestamp","symbol"], how = 'inner').merge(vol, on = ["timestamp","symbol"], how = 'inner').merge(
                                        liquid, on = ['timestamp','symbol'],how = 'inner'
                                    )
        return table
            
        
        