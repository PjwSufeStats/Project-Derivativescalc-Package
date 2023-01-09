'''
File Name : greeks.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This file covers the greeks computation based on black scholes
model and numerical approaches for general greeks computation as well as the 
plots of greeks and the genration of greeks surfaces.
'''

import copy
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from pricing import *
import warnings
warnings.filterwarnings("ignore")


class BlackScholesGreeks(object):
    '''
    compute black scholes greeks
    '''

    def __init__(
        self, 
        spot_price, 
        strike_price, 
        rate, 
        sigma, 
        maturity, 
        option_type='call'):
        '''
        initialize the parameters of calculating black scholes greeks 
        '''

        assert option_type in ('call', 'put'), 'The option type must be call or put'

        self.spot_price = spot_price
        self.strike_price = strike_price
        self.rate = rate
        self.sigma = sigma
        self.tau = maturity
        self.type = option_type

        self.__compute_d()


    def __compute_d(self):
        '''
        compute d(+) and d(-) in black sholes model
        '''

        self.__d_pos = (np.log(self.spot_price / self.strike_price) + ((self.rate + (0.5*self.sigma**2))*self.tau)) / (self.sigma*np.sqrt(self.tau))
        self.__d_neg = (np.log(self.spot_price / self.strike_price) + ((self.rate - (0.5*self.sigma**2))*self.tau)) / (self.sigma*np.sqrt(self.tau))


    def reset_params(func):
        '''
        reset the parameters in the simulation class
        '''

        def wrapper(self, spot_price=None, maturity=None):

            if spot_price != None or maturity != None:
                if spot_price != None:
                    original_param1 = self.spot_price
                    self.spot_price = spot_price
                if maturity != None:
                    original_param2 = self.tau
                    self.tau = maturity
                self.__compute_d()

            #res = func(self, spot_price=None, maturity=None)
            res = func(self, spot_price, maturity)

            if spot_price != None or maturity != None:
                if spot_price != None:
                    self.spot_price = original_param1
                if maturity != None:
                    self.tau = original_param2
                self.__compute_d()

            return res

        return wrapper


    @reset_params
    def delta(self, spot_price=None, maturity=None):
        '''
        compute the delta of black scholes model
        '''

        if self.type == 'call':
            delta_value = norm.cdf(self.__d_pos)
        elif self.type == 'put':
            delta_value = -norm.cdf(-self.__d_pos)

        return delta_value


    @reset_params
    def gamma(self, spot_price=None, maturity=None):
        '''
        compute the gamma of black scholes model
        '''

        gamma_value = norm.pdf(self.__d_pos) / (self.spot_price*self.sigma*np.sqrt(self.tau))

        return gamma_value


    @reset_params
    def vega(self, spot_price=None, maturity=None):
        '''
        compute the vega of black scholes model
        '''

        vega_value = self.spot_price*np.sqrt(self.tau)*norm.pdf(self.__d_pos)

        return vega_value


    @reset_params
    def theta(self, spot_price=None, maturity=None):
        '''
        compute the vega of black scholes model
        '''

        if self.type == 'call':
            theta_value = ((-0.5)*self.sigma*self.spot_price*np.exp((-0.5)*self.__d_pos**2)) / (self.spot_price*self.sigma*np.sqrt(2*np.pi*self.tau)) - (self.rate*self.strike_price*np.exp(-self.rate*self.tau)*norm.cdf(self.__d_neg))
        elif self.type == 'put':
            theta_value = ((-0.5)*self.sigma*self.spot_price*np.exp((-0.5)*self.__d_pos**2)) / (self.spot_price*self.sigma*np.sqrt(2*np.pi*self.tau)) + (self.rate*self.strike_price*np.exp(-self.rate*self.tau)*norm.cdf(-self.__d_neg))

        return theta_value


    @reset_params
    def rho(self, spot_price=None, maturity=None):
        '''
        compute the vega of black scholes model
        '''      

        if self.type == 'call':
            rho_value = self.tau*self.strike_price*np.exp(-self.rate*self.tau)*norm.cdf(self.__d_neg)
        elif self.type == 'put':
            rho_value = -self.tau*self.strike_price*np.exp(-self.rate*self.tau)*norm.cdf(-self.__d_neg)

        return rho_value


    def plot_delta(self, wrt_spot=True, moneyness_cut=0.2):
        '''
        plot delta curve with respect to spot price or maturity
        '''

        if wrt_spot:
            price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), 500)
            delta_range = [self.delta(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Delta - Spot Price Plot of {} Option on Black Scholes Model with Strike Price {}'.format(self.type.upper(), self.strike_price))
            plt.xlabel('Spot Price')
            plt.ylabel('Delta Value')
            plt.grid()
            plt.plot(price_range, delta_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            delta_range = [self.delta(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Delta - Maturity Plot of {} Option on Black Scholes Model with Log Moneyness {}'.format(self.type.upper(), np.log(self.spot_price / self.strike_price)))
            plt.xlabel('Mautiry')
            plt.ylabel('Delta Value')
            plt.grid()
            plt.plot(time_range, delta_range)
            plt.show()


    def plot_gamma(self, wrt_spot=True, moneyness_cut=0.2):
        '''
        plot gamma curve with respect to spot price or maturity
        '''

        if wrt_spot:
            price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), 500)
            gamma_range = [self.gamma(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Gamma - Spot Price Plot on Black Scholes Model with Strike Price {}'.format(self.strike_price))
            plt.xlabel('Spot Price')
            plt.ylabel('Gamma Value')
            plt.grid()
            plt.plot(price_range, gamma_range)
            plt.show()
        else:
            time_range = np.linspace(0.01, 2, 500)
            gamma_range = [self.gamma(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Gamma - Maturity Plot on Black Scholes Model with Log Moneyness {}'.format(np.log(self.spot_price / self.strike_price)))
            plt.xlabel('Mautiry')
            plt.ylabel('Gamma Value')
            plt.grid()
            plt.plot(time_range, gamma_range)
            plt.show()


    def plot_vega(self, wrt_spot=True, moneyness_cut=0.2):
        '''
        plot vega with respect to spot price or maturity
        '''

        if wrt_spot:
            price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), 500)
            vega_range = [self.vega(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Vega - Spot Price Plot on Black Scholes Model with Strike Price {}'.format(self.strike_price))
            plt.xlabel('Spot Price')
            plt.ylabel('Vega Value')
            plt.grid()
            plt.plot(price_range, vega_range)
            plt.show()   
        else:
            time_range = np.linspace(0.01, 2, 500)
            vega_range = [self.vega(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Vega - Maturity Plot of {} Option on Black Scholes Model with Log Moneyness {}'.format(self.type.upper(), np.log(self.spot_price / self.strike_price)))
            plt.xlabel('Mautiry')
            plt.ylabel('Vega Value')
            plt.grid()
            plt.plot(time_range, vega_range)
            plt.show() 


    def plot_theta(self, wrt_spot=True, moneyness_cut=0.2):
        '''
        plot theta with respect to spot price or maturity
        '''

        if wrt_spot:
            price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), 500)
            theta_range = [self.theta(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Theta - Spot Price Plot on Black Scholes Model with Strike Price {}'.format(self.strike_price))
            plt.xlabel('Spot Price')
            plt.ylabel('Theta Value')
            plt.grid()
            plt.plot(price_range, theta_range)
            plt.show()   
        else:
            time_range = np.linspace(0.01, 2, 500)
            theta_range = [self.theta(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Theta - Maturity Plot of {} Option on Black Scholes Model with Log Moneyness {}'.format(self.type.upper(), np.log(self.spot_price / self.strike_price)))
            plt.xlabel('Mautiry')
            plt.ylabel('Theta Value')
            plt.grid()
            plt.plot(time_range, theta_range)
            plt.show()  


    def plot_rho(self, wrt_spot=True, moneyness_cut=0.2):
        '''
        plot rho with respect to spot price or maturity
        '''

        if wrt_spot:
            price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), 500)
            rho_range = [self.rho(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Rho - Spot Price Plot on Black Scholes Model with Strike Price {}'.format(self.strike_price))
            plt.xlabel('Spot Price')
            plt.ylabel('Rho Value')
            plt.grid()
            plt.plot(price_range, rho_range)
            plt.show()   
        else:
            time_range = np.linspace(0.01, 2, 500)
            rho_range = [self.rho(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Rho - Maturity Plot of {} Option on Black Scholes Model with Log Moneyness {}'.format(self.type.upper(), np.log(self.spot_price / self.strike_price)))
            plt.xlabel('Mautiry')
            plt.ylabel('Rho Value')
            plt.grid()
            plt.plot(time_range, rho_range)
            plt.show()  


    def delta_surface(self, moneyness_cut=0.2, interval_t=500, interval_moneyness=500):
        '''
        plot delta surface with respect to log moneyness and maturity
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), interval_moneyness)
        moneyness_range = [np.log(price/self.strike_price) for price in price_range]
        delta_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                delta_array[i, j] = self.delta(spot_price=price_range[j], maturity=time_range[i])

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, delta_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Maturity') 
        ax.set_xlabel('Log Moneyness')    
        ax.set_zlabel('Delta Value')
        plt.title('Delta Surface of {} Option on Black Scholes Model'.format(self.type.upper()))
        plt.show()

        return delta_array


    def gamma_surface(self, moneyness_cut=0.2, interval_t=500, interval_moneyness=500):
        '''
        plot gamma surface with respect to log moneyness and maturity
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), interval_moneyness)
        moneyness_range = [np.log(price/self.strike_price) for price in price_range]
        gamma_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                gamma_array[i, j] = self.gamma(spot_price=price_range[j], maturity=time_range[i])

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, gamma_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Maturity')
        ax.set_xlabel('Log Moneyness')
        ax.set_zlabel('Gamma Value')
        plt.title('Gamma Surface on Black Scholes Model')
        plt.show()

        return gamma_array

 
    def vega_surface(self, moneyness_cut=0.2, interval_t=500, interval_moneyness=500):
        '''
        plot vega surface with respect to log moneyness and maturity
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), interval_moneyness)
        moneyness_range = [np.log(price/self.strike_price) for price in price_range]
        vega_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                vega_array[i, j] = self.vega(spot_price=price_range[j], maturity=time_range[i])

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, vega_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Maturity')
        ax.set_xlabel('Log Moneyness')
        ax.set_zlabel('Vega Value')
        plt.title('Vega Surface of {} Option on Black Scholes Model'.format(self.type.upper()))
        plt.show()

        return vega_array


    def theta_surface(self, moneyness_cut=0.2, interval_t=500, interval_moneyness=500):
        '''
        plot theta surface with respect to log moneyness and maturity
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), interval_moneyness)
        moneyness_range = [np.log(price/self.strike_price) for price in price_range]
        theta_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                theta_array[i, j] = self.theta(spot_price=price_range[j], maturity=time_range[i])

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, theta_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Maturity')
        ax.set_xlabel('Log Moneyness')
        ax.set_zlabel('Theta Value')
        plt.title('Theta Surface of {} Option on Black Scholes Model'.format(self.type.upper()))
        plt.show()

        return theta_array


    def rho_surface(self, moneyness_cut=0.2, interval_t=500, interval_moneyness=500):
        '''
        plot rho surface with respect to log moneyness and maturity
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        price_range = np.linspace(self.strike_price*(1-moneyness_cut), self.strike_price*(1+moneyness_cut), interval_moneyness)
        moneyness_range = [np.log(price/self.strike_price) for price in price_range]
        rho_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                rho_array[i, j] = self.rho(spot_price=price_range[j], maturity=time_range[i])

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, rho_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Maturity')
        ax.set_xlabel('Log Moneyness')
        ax.set_zlabel('Rho Value')
        plt.title('Rho Surface of {} Option on Black Scholes Model'.format(self.type.upper()))
        plt.show()

        return rho_array





class GreeksSim(object):
    '''
    compute greeks numerically
    '''

    def __init__(self, pricing_obj, epsilon=0.01):
        '''
        
        '''

        self.pricing_obj = pricing_obj
        self.epsilon = epsilon


    def reset_epsilon(self, new_epsilon):
        '''
        reset the epsilon
        '''

        self.epsilon = new_epsilon


    def delta(self, spot_price=None, tenor=None):
        '''
        compute the delta numerically
        '''

        temp_pricing_obj = copy.deepcopy(self.pricing_obj)

        if spot_price == None:
            current_price = temp_pricing_obj.spot_price
        else:
            current_price = spot_price
        if tenor != None:
            temp_pricing_obj.price_update_tau(tenor)

        left_value = temp_pricing_obj.price_update_spot(current_price*(1-self.epsilon))
        right_value = temp_pricing_obj.price_update_spot(current_price*(1+self.epsilon))

        del temp_pricing_obj

        if left_value <= 0.0 or right_value <= 0.0:
            return 0
        else:
            return (right_value - left_value) / (2*current_price*self.epsilon)


    def gamma(self, spot_price=None, tenor=None):
        '''
        compute the gamma numerically
        '''

        temp_pricing_obj = copy.deepcopy(self.pricing_obj)

        if spot_price == None:
            current_price = temp_pricing_obj.spot_price
        else:
            current_price = spot_price
        if tenor != None:
            temp_pricing_obj.price_update_tau(tenor)

        left_value = temp_pricing_obj.price_update_spot(current_price*(1-self.epsilon))
        midlle_value = temp_pricing_obj.price_update_spot(current_price)
        right_value = temp_pricing_obj.price_update_spot(current_price*(1+self.epsilon))

        del temp_pricing_obj

        if left_value <= 0.0 or midlle_value <= 0.0 or right_value <= 0.0:
            return 0
        else:
            return (right_value - 2*midlle_value  + left_value) / (current_price*self.epsilon)**2


    def vega(self, spot_price=None, tenor=None):
        '''
        compute the vega numerically
        '''

        temp_pricing_obj = copy.deepcopy(self.pricing_obj)

        if spot_price != None:
            temp_pricing_obj.price_update_spot(spot_price)
        if tenor != None:
            temp_pricing_obj.price_update_tau(tenor)

        try:
            current_sigma = temp_pricing_obj.sigma
        except:
            raise AttributeError('This pricing class has not implemented attribute sigma')

        left_value = temp_pricing_obj.price_update_sigma(current_sigma*(1-self.epsilon))
        right_value = temp_pricing_obj.price_update_sigma(current_sigma*(1+self.epsilon))

        del temp_pricing_obj

        if left_value <= 0.0 or right_value <= 0.0:
            return 0
        else:
            return (right_value - left_value) / (2*current_sigma*self.epsilon)


    def theta(self, spot_price=None, tenor=None):
        '''
        compute the theta numerically
        '''

        temp_pricing_obj = copy.deepcopy(self.pricing_obj)

        if tenor == None:
            current_tenor = temp_pricing_obj.tau
        else:
            current_tenor = tenor
        if spot_price != None:
            temp_pricing_obj.price_update_spot(spot_price)

        left_value = temp_pricing_obj.price_update_tau(current_tenor*(1-self.epsilon))
        right_value = temp_pricing_obj.price_update_tau(current_tenor*(1+self.epsilon))

        del temp_pricing_obj

        if left_value <= 0.0 or right_value <= 0.0:
            return 0
        else:
            return (right_value - left_value) / (2*current_tenor*self.epsilon)


    def rho(self, spot_price=None, tenor=None):
        '''
        compute the rho numerically
        '''

        temp_pricing_obj = copy.deepcopy(self.pricing_obj)

        if spot_price != None:
            temp_pricing_obj.price_update_spot(spot_price)
        if tenor != None:
            temp_pricing_obj.price_update_tau(tenor)

        try:
            current_rate = temp_pricing_obj.rate
        except:
            raise AttributeError('This pricing class has not implemented attribute rate')

        left_value = temp_pricing_obj.price_update_rate(current_rate*(1-self.epsilon))
        right_value = temp_pricing_obj.price_update_rate(current_rate*(1+self.epsilon))

        del temp_pricing_obj

        if left_value <= 0.0 or right_value <= 0.0:
            return 0
        else:
            return (right_value - left_value) / (2*current_rate*self.epsilon)


    def plot_delta(self, wrt_spot=True, moneyness_cut=0.2, interval=50):
        '''
        plot delta curve with respect to spot price or maturity numerically
        '''

        if wrt_spot:
            try:
                price_range = np.linspace(self.pricing_obj.strike_price*(1-moneyness_cut), self.pricing_obj.strike_price*(1+moneyness_cut), interval)
            except:
                price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval)
            delta_range = [self.delta(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Spot Price')
            plt.ylabel('Delta Value')
            plt.grid()
            plt.plot(price_range, delta_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            delta_range = [self.delta(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Mautiry')
            plt.ylabel('Delta Value')
            plt.grid()
            plt.plot(time_range, delta_range)
            plt.show()


    def plot_gamma(self, wrt_spot=True, moneyness_cut=0.2, interval=50):
        '''
        plot gamma curve with respect to spot price or maturity numerically
        '''

        if wrt_spot:
            try:
                price_range = np.linspace(self.pricing_obj.strike_price*(1-moneyness_cut), self.pricing_obj.strike_price*(1+moneyness_cut), interval)
            except:
                price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval)
            gamma_range = [self.gamma(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Spot Price')
            plt.ylabel('Gamma Value')
            plt.grid()
            plt.plot(price_range, gamma_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            gamma_range = [self.gamma(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Mautiry')
            plt.ylabel('Gamma Value')
            plt.grid()
            plt.plot(time_range, gamma_range)
            plt.show()


    def plot_vega(self, wrt_spot=True, moneyness_cut=0.2, interval=50):
        '''
        plot vega curve with respect to spot price or maturity numerically
        '''

        if wrt_spot:
            try:
                price_range = np.linspace(self.pricing_obj.strike_price*(1-moneyness_cut), self.pricing_obj.strike_price*(1+moneyness_cut), interval)
            except:
                price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval)
            vega_range = [self.vega(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Spot Price')
            plt.ylabel('Vega Value')
            plt.grid()
            plt.plot(price_range, vega_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            vega_range = [self.vega(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Mautiry')
            plt.ylabel('Vega Value')
            plt.grid()
            plt.plot(time_range, vega_range)
            plt.show()


    def plot_theta(self, wrt_spot=True, moneyness_cut=0.2, interval=50):
        '''
        plot theta curve with respect to spot price or maturity numerically
        '''

        if wrt_spot:
            try:
                price_range = np.linspace(self.pricing_obj.strike_price*(1-moneyness_cut), self.pricing_obj.strike_price*(1+moneyness_cut), interval)
            except:
                price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval)
            theta_range = [self.theta(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Spot Price')
            plt.ylabel('Theta Value')
            plt.grid()
            plt.plot(price_range, theta_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            theta_range = [self.theta(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Mautiry')
            plt.ylabel('Theta Value')
            plt.grid()
            plt.plot(time_range, theta_range)
            plt.show()


    def plot_rho(self, wrt_spot=True, moneyness_cut=0.2, interval=50):
        '''
        plot rho curve with respect to spot price or maturity numerically
        '''

        if wrt_spot:
            try:
                price_range = np.linspace(self.pricing_obj.strike_price*(1-moneyness_cut), self.pricing_obj.strike_price*(1+moneyness_cut), interval)
            except:
                price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval)
            rho_range = [self.rho(spot_price=price) for price in price_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Spot Price')
            plt.ylabel('Rho Value')
            plt.grid()
            plt.plot(price_range, rho_range)
            plt.show()

        else:
            time_range = np.linspace(0.01, 2, 500)
            rho_range = [self.rho(maturity=time) for time in time_range]

            plt.figure(figsize=(20, 10))
            plt.title('Simulated Delta')
            plt.xlabel('Mautiry')
            plt.ylabel('Rho Value')
            plt.grid()
            plt.plot(time_range, rho_range)
            plt.show()


    def delta_surface(self, moneyness_cut=0.2, interval_t=50, interval_moneyness=50):
        '''
        plot delta surface with respect to log moneyness and maturity numerically
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        try:
            price_range = np.linspace(self.pricing_obj.strike*(1-moneyness_cut), self.pricing_obj.strike*(1+moneyness_cut), interval_moneyness)
            moneyness_range = [np.log(price/self.pricing_obj.strike) for price in price_range]
        except:
            price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval_moneyness)
            moneyness_range = price_range
        delta_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                delta_array[i, j] = self.delta(spot_price=price_range[j], tenor=time_range[i])    

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, delta_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        try:
            self.pricing_obj.strike
            ax.set_xlabel('Log Moneyness')      
        except:   
            ax.set_xlabel('Spot Price')    
        ax.set_ylabel('Maturity')   
        ax.set_zlabel('Delta Value')
        plt.title('Simulated Delta Surface')
        plt.show()

        return delta_array


    def gamma_surface(self, moneyness_cut=0.2, interval_t=50, interval_moneyness=50):
        '''
        plot gamma surface with respect to log moneyness and maturity numerically
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        try:
            price_range = np.linspace(self.pricing_obj.strike*(1-moneyness_cut), self.pricing_obj.strike*(1+moneyness_cut), interval_moneyness)
            moneyness_range = [np.log(price/self.pricing_obj.strike) for price in price_range]
        except:
            price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval_moneyness)
            moneyness_range = price_range
        gamma_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                gamma_array[i, j] = self.gamma(spot_price=price_range[j], tenor=time_range[i])    

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, gamma_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        try:
            self.pricing_obj.strike
            ax.set_xlabel('Log Moneyness')      
        except:   
            ax.set_xlabel('Spot Price')    
        ax.set_ylabel('Maturity')    
        ax.set_zlabel('Gamma Value')
        plt.title('Simulated Gamma Surface')
        plt.show()

        return gamma_array


    def vega_surface(self, moneyness_cut=0.2, interval_t=50, interval_moneyness=50):
        '''
        plot vega surface with respect to log moneyness and maturity numerically
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        try:
            price_range = np.linspace(self.pricing_obj.strike*(1-moneyness_cut), self.pricing_obj.strike*(1+moneyness_cut), interval_moneyness)
            moneyness_range = [np.log(price/self.pricing_obj.strike) for price in price_range]
        except:
            price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval_moneyness)
            moneyness_range = price_range
        vega_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                vega_array[i, j] = self.vega(spot_price=price_range[j], tenor=time_range[i])    

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, vega_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        try:
            self.pricing_obj.strike
            ax.set_xlabel('Log Moneyness')      
        except:   
            ax.set_xlabel('Spot Price')    
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Vega Value')
        plt.title('Simulated Vega Surface')
        plt.show()

        return vega_array


    def theta_surface(self, moneyness_cut=0.2, interval_t=50, interval_moneyness=50):
        '''
        plot theta surface with respect to log moneyness and maturity numerically
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        try:
            price_range = np.linspace(self.pricing_obj.strike*(1-moneyness_cut), self.pricing_obj.strike*(1+moneyness_cut), interval_moneyness)
            moneyness_range = [np.log(price/self.pricing_obj.strike) for price in price_range]
        except:
            price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval_moneyness)
            moneyness_range = price_range
        theta_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                theta_array[i, j] = self.theta(spot_price=price_range[j], tenor=time_range[i])    

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, theta_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        try:
            self.pricing_obj.strike
            ax.set_xlabel('Log Moneyness')      
        except:   
            ax.set_xlabel('Spot Price')  
        ax.set_zlabel('Theta Value')
        plt.title('Simulated Theta Surface')
        plt.show()

        return theta_array


    def rho_surface(self, moneyness_cut=0.2, interval_t=50, interval_moneyness=50):
        '''
        plot rho surface with respect to log moneyness and maturity numerically
        '''

        time_range = np.linspace(0.01, 2, interval_t)
        try:
            price_range = np.linspace(self.pricing_obj.strike*(1-moneyness_cut), self.pricing_obj.strike*(1+moneyness_cut), interval_moneyness)
            moneyness_range = [np.log(price/self.pricing_obj.strike) for price in price_range]
        except:
            price_range = np.linspace(self.pricing_obj.spot_price*(1-moneyness_cut), self.pricing_obj.spot_price*(1+moneyness_cut), interval_moneyness)
            moneyness_range = price_range
        rho_array = np.zeros((len(time_range), len(moneyness_range)))

        for i in range(len(time_range)):
            for j in range(len(moneyness_range)):
                rho_array[i, j] = self.rho(spot_price=price_range[j], tenor=time_range[i])    

        fig = plt.figure(figsize=(20,20))
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(moneyness_range, time_range)
        surf = ax.plot_surface(X, Y, rho_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        try:
            self.pricing_obj.strike
            ax.set_xlabel('Log Moneyness')      
        except:   
            ax.set_xlabel('Spot Price')     
        ax.set_zlabel('Rho Value')
        plt.title('Simulated Rho Surface')
        plt.show()

        return rho_array

