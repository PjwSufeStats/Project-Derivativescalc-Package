'''
File Name : volatility.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This file covers the implied volatility computation
and the generation of volatility surfaces.
'''


from pricing import black_scholes
from greeks import BlackScholesGreeks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def implied_vol(
    spot_price,
    strike_price,
    target_price,
    rate,
    maturity, 
    option_type='call', 
    accuracy=1e-4):
    '''
    calculate implied volatility with newton method
    '''
    
    assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

    vol = 0.2
    diff = np.inf
    n_iter = 0

    while abs(diff) > accuracy:

        new_price = black_scholes(spot_price, strike_price, rate, vol, maturity, option_type.lower())
        new_vega = BlackScholesGreeks(spot_price, strike_price, rate, vol, maturity, option_type.lower()).vega()

        diff = target_price - new_price
        vol += diff / new_vega
        n_iter += 1

        if n_iter > 1000:
            raise ValueError('Interation times over limit')

    return vol





class VolSurface(object):
    '''
    class for volatility surface generation
    '''

    def __init__(
        self,
        ticker,
        date,
        spot_price,
        maturity_list,
        strike_list,
        option_prices_arr,   
        rate=0,
        option_type='call',
        accuracy=1e-4):
        '''
        intiaialize the model parameters
        '''

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.ticker = ticker
        self.date = date
        self.spot_price = float(spot_price)
        self.maturity_list = list(map(float, list(maturity_list)))   
        self.strike_list = list(map(float, list(strike_list)))     
        self.option_prices_arr = np.array(option_prices_arr)
        self.rate = rate
        self.option_type = option_type.lower()
        self.accuracy = accuracy


    def implied_vol(
        self,
        maturity,
        strike_price,
        target_price):
        '''
        calculate implied volatility
        '''

        vol = 0.5
        diff = np.inf
        n_iter = 0

        while abs(diff) > self.accuracy:

            new_price = black_scholes(self.spot_price, strike_price, self.rate, vol, maturity, self.option_type)
            new_vega = BlackScholesGreeks(self.spot_price, strike_price, self.rate, vol, maturity, self.option_type).vega()

            diff = target_price - new_price
            vol += diff / new_vega
            n_iter += 1

            if n_iter > 1000:
                raise ValueError('Interation times over limit')

        return vol


    def get_vol_matrix(self):
        '''
        get volatility matrix
        '''

        self.vol_matrix = np.zeros((len(self.maturity_list), len(self.strike_list)))
        for i in range(len(self.maturity_list)):
            for j in range(len(self.strike_list)):
                if self.option_prices_arr[i, j] != 0 and not np.isnan(self.option_prices_arr[i, j]) and self.maturity_list[i] > 0:
                    self.vol_matrix[i, j] = self.implied_vol(float(self.maturity_list[i]), float(self.strike_list[j]), float(self.option_prices_arr[i, j]))
                else:
                    self.vol_matrix[i, j] = np.nan

        return self.vol_matrix

    def plot_surface(self):
        '''
        plot volatility surface
        '''

        self.get_vol_matrix()

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(self.strike_list, self.maturity_list)
        surf = ax.plot_surface(X, Y, self.vol_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        plt.title('Implied Volatility of {} Option Spot on {} in {}'.format(self.option_type.upper(), self.ticker, self.date))
        plt.show()