'''
File Name : pricing.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This file covers the fundamental pricing approaches for different derivatives structures,
The monte carlo simulation approach, binominal trees approach and finite difference approach has been
introduced into pricing. The monte carlo approach allows user to select different underlying dynamics
in paths.py to generate paths. Futhermore, this file also contains the pricing approaches for some popular 
OTC derivatives such as snow ball product.
'''

import abc
from scipy.stats import norm
from paths import * 
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor





def forward(
    spot_price,
    rate,
    maturity):
    '''
    forward theoretical price with constant discounted rates
    '''

    return np.exp(rate*maturity) * spot_price



def black_scholes(
    spot_price,
    strike_price, 
    rate,
    sigma, 
    maturity,
    option_type='call',
    dividend=0):
    '''
    Black Scholes analytical price for european vanilla options
    from Black and Scholes, 1973
    '''

    assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

    d_pos = (np.log(spot_price / strike_price) + ((rate - dividend + (0.5*sigma**2))*maturity)) / (sigma*np.sqrt(maturity))
    d_neg = (np.log(spot_price / strike_price) + ((rate - dividend - (0.5*sigma**2))*maturity)) / (sigma*np.sqrt(maturity))

    if option_type == 'call':
        price = (spot_price*norm.cdf(d_pos)) - (strike_price*np.exp(-(rate-dividend)*maturity)*norm.cdf(d_neg))
    elif option_type == 'put':
        price = (strike_price*np.exp(-(rate-dividend)*maturity)*norm.cdf(-d_neg)) - (spot_price*norm.cdf(-d_pos))

    return price



def knock_out_barrier_analytical(
    spot_price, 
    strike_price, 
    barrier,
    rate, 
    sigma, 
    maturity, 
    option_type='call'):
    '''
    analytical price of knock out barrier option
    from Rubinstein and Reiner, 1993
    '''

    assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

    def delta_func(pos_or_neg, tau, u):

        if pos_or_neg:
            return (1/(sigma*np.sqrt(tau))) * (np.log(u) + (rate+0.5*sigma**2)*tau)
        else:
            return (1/(sigma*np.sqrt(tau))) * (np.log(u) + (rate-0.5*sigma**2)*tau)

    I1 = norm.cdf(delta_func(True, maturity, spot_price/strike_price)) - norm.cdf(delta_func(True, maturity, spot_price/barrier))
    I2 = np.exp(rate*maturity) * (norm.cdf(delta_func(False, maturity, spot_price/strike_price)) - norm.cdf(delta_func(False, maturity, spot_price/barrier)))
    I3 = (spot_price/barrier)**(-(2*rate/sigma**2)-1) * (norm.cdf(delta_func(True, maturity, barrier**2/(strike_price*spot_price))) - norm.cdf(delta_func(True, maturity, barrier/spot_price)))
    I4 = np.exp(-rate*maturity) * (spot_price/barrier)**(-(2*rate/sigma**2)+1) * (norm.cdf(delta_func(False, maturity, barrier**2/(strike_price*spot_price))) - norm.cdf(delta_func(False, maturity, barrier/spot_price)))

    call_price = spot_price*I1 - strike_price*I2 - spot_price*I3 + strike_price*I4

    if option_type == 'call':
        return call_price
    elif option_type == 'put':
        return call_price + strike_price*np.exp(-rate*maturity) - spot_price








class Price(metaclass=abc.ABCMeta):
    '''
    Price abstract base class which should only be inherited by subclasses
    '''

    def __init__(
        self,
        init_price, 
        maturity):
        '''
        intiaialize the structure parameters
        '''

        self.spot_price = init_price
        self.tau = maturity


    @abc.abstractmethod
    def value(self):
        '''
        valuation of the derivative
        '''

        print('Base class Price has no concrete implementation of .value() and return None')

        price_value = None

        return price_value


    def reset_param(self, param, new_value):
        '''
        reset the parameters in the simulation class
        '''

        if param == 'spot_price':
            self.spot_price = new_value
        elif param == 'tau':
            self.tau = new_value
            try:
                self.n_intervals = self.nper_per_year * self.tau
            except:
                pass
        elif param == 'sigma' :
            self.sigma = new_value
        elif param == 'rate':
            self.rate = new_value
        else:
            raise NameError('The parameter to be modified is not correct')


    def price_update_spot(self, new_spot=None):
        '''
        recompute the derivatives value with new spot price
        '''

        self.reset_param('spot_price', new_spot)

        return self.value()


    def price_update_tau(self, new_maturity=None):
        '''
        recompute the derivatives value with new tenor
        '''

        self.reset_param('tau', new_maturity)

        return self.value()


    def price_update_sigma(self, new_sigma=None):
        '''
        recompute the derivatives value with new volatility
        '''

        self.reset_param('sigma', new_sigma)

        return self.value()


    def price_update_rate(self, new_rate=None):
        '''
        recompute the derivatives value with new rate
        '''

        self.reset_param('rate', new_rate)

        return self.value()





class MonteCarloSim(Price):
    '''
    Monte Carlo simulation abstract base class which should only be inherited by subclasses
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        model, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity)

        model_types_list = [
            CEV, 
            GeoBrownianMotion, 
            Bachelier, 
            OU, 
            CIR, 
            SABR, 
            Heston,
            Merton,
            Bates]
        assert model in model_types_list, 'Selcet an available model type'

        self.n_trials = n_trials
        self.nper_per_year = nper_per_year
        self.n_intervals = int(nper_per_year * self.tau)
        self.model_params = model_params
        self.model = model

        if 'rate' in model_params.keys():
            self.rate = model_params['rate']
        elif 'initialized_rate' in model_params.keys():
            self.rate = model_params['initialized_rate']
        if 'sigma' in model_params.keys():
            self.sigma = model_params['sigma']
        elif 'initialized_volatility' in model_params.keys():
            self.sigma = model_params['initialized_volatility']


    def generate_paths(func):
        '''
        paths generate decorator, the overwritten value method must be decorated by this decorator
        in order to generate paths again when the parameters may be changed
        '''

        def wrapper(self):

            path = self.model(self.n_trials, self.n_intervals, **self.model_params)
            self.price_process = path.generate_paths()

            return func(self)

        return wrapper


    @abc.abstractmethod
    @generate_paths
    def value(self):
        '''
        valuation of the derivatives
        abstract method which must be implemented in the inherited class
        the overwritten method of 'value()' in the inherited class must be decorated with 'MonteCarloSim.generate_paths' decorator
        '''

        print('Base class MonteCarloSim has no concrete implementation of .value() and return None')

        self.payoff_vector = None
        simulated_price = None

        return simulated_price


    def std(self):
        '''
        compute standard variation of the sample paths
        '''

        try:
            return np.std(self.payoff_vector)
        except:
            raise ValueError('There is no payoff which is computed')


    def reset_param(self, param, new_value):
        '''
        reset the parameters in the simulation class
        '''

        assert param in self.model_params.keys(), 'The parameter to be modified is not correct'

        self.model_params[param] = new_value
        
        if param == 'spot_price':
            self.spot_price = new_value
        elif param == 'tau':
            self.tau = new_value
            self.n_intervals = int(self.nper_per_year * self.tau)
        elif param == 'sigma' or param == 'initialized_volatility':
            try:
                self.sigma = new_value
            except:
                raise AttributeError('This pricing class has not implemented attribute sigma')
        elif param == 'rate' or param == 'initialized_rate':
            try:
                self.rate = new_value
            except:
                raise AttributeError('This pricing class has not implemented attribute rate')


    def price_update_spot(self, new_spot):
        '''
        recompute the derivatives value with new spot price
        '''

        self.reset_param('spot_price', new_spot)

        return self.value()


    def price_update_tau(self, new_maturity):
        '''
        recompute the derivatives value with new tenor
        '''

        self.reset_param('tau', new_maturity)

        return self.value()


    def price_update_sigma(self, new_sigma):
        '''
        recompute the derivatives value with new volatility
        '''

        if 'sigma' in self.model_params.keys():
            self.reset_param('sigma', new_sigma)
        elif 'initialized_volatility' in self.model_params.keys():
            self.reset_param('initialized_volatility', new_sigma)
        else:
            raise NameError('This model does not have a volatility related parameter')

        return self.value()


    def price_update_rate(self, new_rate):
        '''
        recompute the derivatives value with new rate
        '''

        if 'rate' in self.model_params.keys():
            self.reset_param('rate', new_rate)
        elif 'initialized_rate' in self.model_params.keys():
            self.reset_param('rate', new_rate)
        else:
            raise NameError('This model does not have a interest rate related parameter')

        return self.value()





class EuropeanVanillaSim(MonteCarloSim):
    '''
    monte carlo simulation for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        model=GeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        self.payoff_vector = relu_func(self.price_process[:,-1])
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price



class EuropeanBarrierSim(MonteCarloSim):
    '''
    monte carlo simulation for european barrier options
    '''

    def __init__(
        self,
        init_price, 
        maturity, 
        n_trials,
        nper_per_year, 
        strike, 
        barrier_up=None, 
        barrier_down=None,
        option_type='call', 
        knock_type='out', 
        direction='up', 
        model=GeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert knock_type.lower() in ('in', 'out'), 'Knock type must be in or out'
        assert direction.lower() in ('up', 'down', 'double'), 'Direction type must be up or down or double'
        if direction == 'up':
            assert barrier_up != None , 'Please input an upward barrier'
        elif direction == 'down':
            assert barrier_down != None , 'Please input a downward barrier'
        elif direction == 'double':
            assert barrier_up != None and barrier_down != None, 'Please input both an upward barrier and a downward barrier'

        self.strike = strike
        self.option_type = option_type
        self.knock_type = knock_type
        self.direction = direction
        self.barrier_up = init_price*barrier_up
        self.barrier_down = init_price*barrier_down


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        indicator_matrix = np.zeros((self.n_trials,1))

        if self.direction == 'up':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.max(self.price_process,axis=1) < self.barrier_up, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = np.array(np.max(self.price_process,axis=1) >= self.barrier_up, dtype=int)
        elif self.direction == 'down':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.min(self.price_process,axis=1) > self.barrier_down, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = np.array(np.min(self.price_process,axis=1) <= self.barrier_down, dtype=int)
        elif self.direction == 'double':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.max(self.price_process,axis=1) < self.barrier_up, dtype=int) * np.array(np.min(self.price_process,axis=1) > self.barrier_down, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = 1-((1-np.array(np.max(self.price_process,axis=1) >= self.barrier_up, dtype=int)) * (1-np.array(np.min(self.price_process,axis=1) <= self.barrier_down, dtype=int)))

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)  

        self.payoff_vector = relu_func(self.price_process[:,-1])*indicator_matrix
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class EuropeanLookbackSim(MonteCarloSim):
    '''
    monte carlo simulation for european lookback options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year,
        fixed_strike=False, 
        strike=None,
        option_type='call', 
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        if fixed_strike == True:
            assert strike != None, 'The srike should be give if this is a fixed strike lookback option'
            self.strike = strike

        self.option_type = option_type
        self.fixed_strike = fixed_strike


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1) 

        if self.fixed_strike:
            if self.option_type == 'call':
                self.payoff_vector = relu_func(np.max(self.price_process, axis=1))
            elif self.option_type == 'put':
                self.payoff_vector = relu_func(np.min(self.price_process, axis=1))

        else:
            if self.option_type == 'call':
                self.payoff_vector = self.price_process[:,-1] - np.min(self.price_process, axis=1)
            elif self.option_type == 'put':
                self.payoff_vector = np.max(self.price_process, axis=1) - self.price_process[:,-1]

        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price 





class EuropeanAsianSim(MonteCarloSim):
    '''
    monte carlo simulation for european asian options
    '''

    def __init__(
        self,
        init_price, 
        maturity,
        n_trials,
        nper_per_year, 
        strike,
        option_type='call',
        ave_type='arith', 
        model=GeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert ave_type.lower() in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1) 

        if self.ave_type == 'arith':
            self.payoff_vector = relu_func(np.mean(self.price_process, axis=1))
        elif self.ave_type == 'geo':
            self.payoff_vector = relu_func(np.exp(np.mean(np.log(self.price_process), axis=1)))

        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class EuropeanBinarySim(MonteCarloSim):
    '''
    monte carlo simulation for european binary options
    '''

    def __init__(
        self, 
        init_price,
        maturity, 
        n_trials,
        nper_per_year, 
        strike,
        option_type='call', 
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:1 if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:1 if self.strike-x >= 0 else 0,1,1)

        self.payoff_vector = relu_func(self.price_process[:,-1])
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price




class ChooserSim(MonteCarloSim):
    '''
    monte carlo simulation for chooser options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        choose_time,
        model=GeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert 0 < choose_time < 1, 'The choose time must be between 0 and 1'

        self.strike = strike
        self.choose_time = choose_time
        self.choose_index = int(choose_time*self.n_intervals)


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        relu_call_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        relu_put_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        self.payoff_vector = np.exp(-self.rate*self.tau)*relu_call_func(self.price_process[:,-1]) + np.exp(-self.rate*self.choose_time*self.tau)*relu_put_func(self.price_process[:,self.choose_index])
        simulated_price = np.mean(self.payoff_vector)

        return simulated_price





class EuropeanParisianSim(EuropeanBarrierSim):
    '''
    monte carlo simulation for european parisian options
    '''

    def __init__(
        self,
        init_price,
        maturity, 
        n_trials, 
        nper_per_year,
        strike,
        observ_window,
        barrier_up=None, 
        barrier_down=None,
        option_type='call', 
        knock_type='out', 
        direction='up', 
        observ_type='continous',
        model=GeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        assert observ_type in ('continous', 'accum'), 'The observation type must be continous or accum'

        super().__init__(init_price, maturity, n_trials, nper_per_year, strike, barrier_up, barrier_down, option_type, knock_type, direction, model, **model_params)

        self.observ_window = observ_window
        self.observ_type = observ_type


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''        

        indicator_matrix = np.zeros((self.n_trials,1))

        if self.observ_type == 'continous':
            
            conv_matrix = np.array([[(lambda _: 1 if j >= i and j < i+self.observ_window else 0)(j) for j in range(self.n_intervals)] for i in range(self.n_intervals-self.observ_window+1)]).T

            if self.direction == 'up':
                if self.knock_type == 'out':
                    indicator_matrix = 1 - np.max(np.where((np.where(self.price_process >= self.barrier_up, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)
                elif self.knock_type == 'in':
                    indicator_matrix = np.max(np.where((np.where(self.price_process >= self.barrier_up, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)
            elif self.direction == 'down':
                if self.knock_type == 'out':
                    indicator_matrix = 1 - np.max(np.where((np.where(self.price_process <= self.barrier_down, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)
                elif self.knock_type == 'in':
                    indicator_matrix = np.max(np.where((np.where(self.price_process <= self.barrier_down, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)
            elif self.direction == 'double':
                if self.knock_type == 'out':
                    indicator_matrix = ((1 - np.max(np.where((np.where(self.price_process >= self.barrier_up, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)) *
                                        (1 - np.max(np.where((np.where(self.price_process <= self.barrier_down, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)))
                elif self.knock_type == 'in':
                    indicator_matrix = 1 - ((1 - np.max(np.where((np.where(self.price_process >= self.barrier_up, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)) *
                                        (1 - np.max(np.where((np.where(self.price_process <= self.barrier_down, 1, 0) @ conv_matrix) >= self.observ_window, 1, 0), axis=1)))     

        elif self.observ_type == 'accum':

            if self.direction == 'up':
                if self.knock_type == 'out':
                    indicator_matrix = np.array(np.sum(np.where(self.price_process >= self.barrier_up, 1, 0), axis=1) < self.observ_window, dtype=int)
                elif self.knock_type == 'in':
                    indicator_matrix = np.array(np.sum(np.where(self.price_process >= self.barrier_up, 1, 0), axis=1) >= self.observ_window, dtype=int)
            elif self.direction == 'down':
                if self.knock_type == 'out':
                    indicator_matrix = np.array(np.sum(np.where(self.price_process <= self.barrier_down, 1, 0), axis=1) < self.observ_window, dtype=int)
                elif self.knock_type == 'in':
                    indicator_matrix = np.array(np.sum(np.where(self.price_process <= self.barrier_down, 1, 0), axis=1) >= self.observ_window, dtype=int)
            elif self.direction == 'double':
                if self.knock_type == 'out':
                    indicator_matrix = (np.array(np.sum(np.where(self.price_process >= self.barrier_up, 1, 0), axis=1) < self.observ_window, dtype=int) *
                                        np.array(np.sum(np.where(self.price_process <= self.barrier_down, 1, 0), axis=1) < self.observ_window, dtype=int))
                elif self.knock_type == 'in':
                    indicator_matrix = 1 - ((1-np.array(np.sum(np.where(self.price_process >= self.barrier_up, 1, 0), axis=1) >= self.observ_window, dtype=int)) *
                                        (1-np.array(np.sum(np.where(self.price_process <= self.barrier_down, 1, 0), axis=1) >= self.observ_window, dtype=int)))
                    
        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)  

        self.payoff_vector = relu_func(self.price_process[:,-1])*indicator_matrix
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class SnowBallSim(MonteCarloSim):
    '''
    monte carlo simulation for snow ball structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        fixed_coupond,
        ki_level,
        ko_level,
        step_down=0,
        ki_fre='day',
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        fre_dict = {'day':1, 'month':30, 'year':360, None:None}
        assert ki_fre in fre_dict.keys() and ko_fre in fre_dict.keys() and ko_fre in fre_dict.keys(), 'The input knock in or knock out frequency is not valid'

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        self.rate = model_params['rate']
        self.notional_principal = notional_principal
        self.fixed_coupond = fixed_coupond
        self.ki_barrier = init_price * ki_level
        self.ko_barrier = init_price * ko_level
        self.step_down = step_down
        self.ki_interval = fre_dict[ki_fre]
        self.ko_interval = fre_dict[ko_fre]
        self.sd_interval = fre_dict[sd_fre]
        self.skip_ko_observe = skip_ko_observe


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ki_indicator_matrix = (np.where(self.price_process < self.ki_barrier, 1, 0) * np.array([[(lambda x:1 if (x+1)%self.ki_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ki_indicator = np.max(ki_indicator_matrix, axis=1)
        ko_indicator = np.max(ko_indicator_matrix, axis=1)

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if self.spot_price-x >= 0 else 0,1,1)    
        ki_payoff = relu_func(self.price_process[:,-1]) * ki_indicator
        ko_payoff = (np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator * (self.fixed_coupond/self.nper_per_year)
        nki_nko_payoff = self.fixed_coupond*self.tau * (1 - ki_indicator) * (1 - ko_indicator)
        self.payoff_vector = self.notional_principal * (ko_payoff + ki_payoff*(1 - ko_indicator) + nki_nko_payoff)

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price





class ParisianSnowBallSim(SnowBallSim):
    '''
    monte carlo simulation for parisian snow ball structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        fixed_coupond,
        ki_level,
        ko_level,
        observ_window,
        step_down=0,
        ki_fre='day',
        ko_fre='month',
        sd_fre='month',
        observ_type='continous',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, notional_principal, fixed_coupond, ki_level, ko_level,
                        step_down, ki_fre, ko_fre, sd_fre, skip_ko_observe, model, **model_params)

        assert observ_type in ('continous', 'accum'), 'The observation type must be continous or accum'

        self.observ_window = observ_window
        self.observ_type = observ_type


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ki_indicator_matrix = (np.where(self.price_process < self.ki_barrier, 1, 0) * np.array([[(lambda x:1 if (x+1)%self.ki_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        
        if self.observ_type == 'continous':
            conv_matrix = np.array([[(lambda _: 1 if j >= i*self.ki_interval and j < (i+self.observ_window)*self.ki_interval and (j+1)%self.ki_interval == 0 else 0)(j) for j in range(self.n_intervals)] for i in range(int(self.n_intervals/self.ki_interval)-self.observ_window+1)]).T 
            ki_indicator = np.max(np.where((ki_indicator_matrix @ conv_matrix) >= self.observ_window, 1, 0), axis=1)
        elif self.observ_type == 'accum':
            ki_indicator = np.max(np.where(np.sum(ki_indicator_matrix, axis=1) >= self.observ_window, 1, 0), axis=1)
        ko_indicator = np.max(ko_indicator_matrix, axis=1)

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if self.spot_price-x >= 0 else 0,1,1)    
        ki_payoff = relu_func(self.price_process[:,-1]) * ki_indicator
        ko_payoff = (np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator * (self.fixed_coupond/self.nper_per_year)
        nki_nko_payoff = self.fixed_coupond*self.tau * (1 - ki_indicator) * (1 - ko_indicator)
        self.payoff_vector = self.notional_principal * (ko_payoff + ki_payoff*(1 - ko_indicator) + nki_nko_payoff)

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price





class PhoenixSim(SnowBallSim):
    '''
    monte carlo simulation for phoenix structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        fixed_coupond,
        ki_level,
        ko_level,
        step_down=0,
        ki_fre='day',
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, notional_principal, fixed_coupond, ki_level, ko_level,
                        step_down, ki_fre, ko_fre, sd_fre, skip_ko_observe, model, **model_params)

    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ki_indicator_matrix = (np.where(self.price_process < self.ki_barrier, 1, 0) * np.array([[(lambda x:1 if (x+1)%self.ki_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ki_indicator = np.max(ki_indicator_matrix, axis=1)
        ko_indicator = np.max(ko_indicator_matrix, axis=1)
        
        yield_payoff = (1 - ki_indicator_matrix) * np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials)
        ko_yield = yield_payoff * np.where(np.array([[i+1 for i in range(self.n_intervals)]]*self.n_trials) < np.array((np.argmax(ko_indicator_matrix, axis=1)+1)*ko_indicator).reshape(self.n_trials,1), 1, 0)  
        ko_payoff = np.sum(ko_yield, axis=1) * ((self.fixed_coupond*self.ko_interval)/self.nper_per_year)

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if self.spot_price-x >= 0 else 0,1,1)    
        ki_payoff = (relu_func(self.price_process[:,-1]) + (np.sum(yield_payoff, axis=1) * ((self.fixed_coupond*self.ko_interval)/self.nper_per_year))) * ki_indicator

        nki_nko_payoff = self.fixed_coupond*self.tau * (1 - ki_indicator) * (1 - ko_indicator)
        self.payoff_vector = self.notional_principal * (ko_payoff + ki_payoff*(1 - ko_indicator) + nki_nko_payoff)

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price




class FCNSim(SnowBallSim):
    '''
    monte carlo simulation for FCN structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        fixed_coupond,
        ki_level,
        ko_level,
        step_down=0,
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, notional_principal, fixed_coupond, ki_level, ko_level, 
                        step_down, None, ko_fre, sd_fre, skip_ko_observe, model, **model_params)

    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''
        
        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator = np.max(ko_indicator_matrix, axis=1)

        ko_payoff = (np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator * (self.fixed_coupond/self.nper_per_year)

        relu_func = np.frompyfunc(lambda x:(x/self.ki_barrier)-1 if self.ki_barrier-x >= 0 else 0,1,1) 
        no_ki_payoff = (relu_func(self.price_process[:,-1]) + (self.fixed_coupond*self.tau))

        self.payoff_vector = self.notional_principal * (ko_payoff*ko_indicator + no_ki_payoff*(1-ko_indicator))

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price





class AirBagSim(MonteCarloSim):
    '''
    monte carlo simulation for air bag structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        ki_level,
        ki_participate,
        no_ki_participate,
        ki_fre='day',
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        fre_dict = {'day':1, 'month':30, 'year':360}
        assert ki_fre in fre_dict.keys(), 'The input knock in frequency is not valid'

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        self.rate = model_params['rate']
        self.notional_principal = notional_principal
        self.ki_barrier = init_price * ki_level
        self.ki_participate = ki_participate
        self.no_ki_participate = no_ki_participate
        self.ki_interval = fre_dict[ki_fre]


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ki_indicator_matrix = (np.where(self.price_process < self.ki_barrier, 1, 0) * np.array([[(lambda x:1 if (x+1)%self.ki_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ki_indicator = np.max(ki_indicator_matrix, axis=1)
        
        ki_payoff_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 ,1,1)    
        ki_payoff = self.ki_participate * ki_payoff_func(self.price_process[:,-1]) * ki_indicator

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if x-self.spot_price >= 0 else 0,1,1)
        no_ki_payoff = self.no_ki_participate * relu_func(self.price_process[:,-1]) 

        self.payoff_vector = self.notional_principal * (no_ki_payoff*(1-ki_indicator) + ki_payoff*ki_indicator)
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class BoosterSim(SnowBallSim):
    '''
    monte carlo simulation for booster structures
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_trials,
        nper_per_year,
        notional_principal,
        fixed_coupond,
        ki_level,
        ko_level,
        participate,
        step_down=0,
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        assert ko_level > 1, 'The knock level must be greater than 100%'
        assert ki_level < 1, 'The knock level must be less than 100%'

        super().__init__(init_price, maturity, n_trials, nper_per_year, notional_principal, fixed_coupond, ki_level, ko_level,
                        step_down, None, ko_fre, sd_fre, skip_ko_observe, model, **model_params)

        self.participate = participate


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator = np.max(ko_indicator_matrix, axis=1)

        ko_payoff = self.fixed_coupond

        relu_func = np.frompyfunc(lambda x:self.participate*((x/self.spot_price)-1) if x-self.spot_price >= 0 else (self.ki_barrier/self.spot_price)-1 if x-self.ki_barrier <= 0 else (x/self.spot_price)-1 ,1,1)
        no_ko_payoff = relu_func(self.price_process[:,-1])

        self.payoff_vector = self.notional_principal * (ko_payoff*ko_indicator + no_ko_payoff*(1-ko_indicator))

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price






class BinomialTree(Price):
    '''
    binomial tree pricing model base class
    '''

    def __init__(
        self,
        init_price,
        maturity,
        rate,
        sigma,
        nper_per_year,
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity)

        self.rate = rate
        self.sigma = sigma
        self.dividend = dividend
        self.nper_per_year = nper_per_year
        self.delta_time = 1 / nper_per_year
        self.n_intervals = int(nper_per_year * self.tau)

        self.u = np.exp(self.sigma*np.sqrt(self.delta_time))
        self.d = np.exp(-self.sigma*np.sqrt(self.delta_time))
        self.a = np.exp((self.rate-self.dividend)*self.delta_time)
        self.p_tilde = (self.a - self.d)/(self.u - self.d)
        self.q_tilde = 1 - self.p_tilde


    @abc.abstractmethod
    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        return None


    def on_path_update(self, spot_price, derivative_value):
        '''
        update the derivatives prices on each node
        '''

        return None


    def recursion(self, n_iter, spot_price):
        '''
        binomial tree recursion algorithm
        '''

        if n_iter > 1:

            derivative_value = np.exp(-(self.rate-self.dividend)*self.delta_time) * (self.p_tilde*self.recursion(n_iter-1, self.u*spot_price) + self.q_tilde*self.recursion(n_iter-1, self.d*spot_price))
            updated_value = self.on_path_update(spot_price, derivative_value)

            if updated_value != None:
                return updated_value
            else:
                return derivative_value

        else:
            return self.payoff(spot_price)

    def value(self):
        '''
        valuation of the derivative
        '''

        return self.recursion(self.n_intervals, self.spot_price)





class EuropeanVanillaBinomial(BinomialTree):
    '''
    binomial tree pricing for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        nper_per_year, 
        strike, 
        option_type='call'):
        '''
        intiaialize the model parameters
        '''
        
        super().__init__(init_price, maturity, rate, sigma, nper_per_year)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type


    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0





class EuropeanAsianBinomial(BinomialTree):
    '''
    binominal tree pricing for european asian options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        nper_per_year, 
        strike, 
        option_type='call', 
        ave_type='arith'):
        '''
        intiaialize the model parameters
        '''
        
        super().__init__(init_price, maturity, rate, sigma, nper_per_year)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert ave_type.lower() in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type


    def payoff(self, on_path_sum):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.ave_type == 'arith':
            ave_price = on_path_sum / self.n_intervals
        elif self.ave_type == 'geo':
            ave_price = on_path_sum ** (1/self.n_intervals)

        if self.option_type == 'call':
            return ave_price - self.strike if ave_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - ave_price if ave_price - self.strike < 0 else 0


    def recursion(self, n_iter, spot_price, on_path_sum=0):
        '''
        recursion algorithm for binomial tree pricing model
        '''

        if self.ave_type == 'arith':
            on_path_sum += spot_price
        elif self.ave_type == 'geo':
            on_path_sum *= spot_price
        
        if n_iter > 1:

            derivative_value = np.exp(-(self.rate-self.dividend)*self.delta_time) * (self.p_tilde*self.recursion(n_iter-1, self.u*spot_price, on_path_sum) + self.q_tilde*self.recursion(n_iter-1, self.d*spot_price, on_path_sum))
            updated_value = self.path_judgement(spot_price, derivative_value)

            if updated_value != None:
                return updated_value
            else:
                return derivative_value
        
        else:
            return self.payoff(on_path_sum)




class AmericanVanillaBinomial(BinomialTree):
    '''
    binomial tree pricing for american vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        nper_per_year, 
        strike, 
        option_type='call'):
        '''
        intiaialize the model parameters
        '''
        
        super().__init__(init_price, maturity, rate, sigma, nper_per_year)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        
        self.strike = strike
        self.option_type = option_type


    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0


    def on_path_update(self, spot_price, derivative_value):
        '''
        update the derivatives prices on each node
        '''

        if self.option_type == 'call' and spot_price - self.strike > derivative_value:
            return spot_price - self.strike
        elif self.option_type == 'put' and self.strike - spot_price > derivative_value:
            return self.strike - spot_price
        else:
            return None





class ShoutBinomial(BinomialTree):
    '''
    binomial tree pricing for shout options
    '''

    def __init__(
        self,
        init_price,
        maturity,
        rate,
        sigma,
        nper_per_year, 
        strike,
        option_type='call'):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, rate, sigma, nper_per_year)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        
        self.strike = strike
        self.option_type = option_type


    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0


    def on_path_update(self, spot_price, derivative_value):
        '''
        update the derivatives prices on each node
        '''

        if self.option_type == 'call' and spot_price - self.strike > derivative_value:
            return spot_price - self.strike
        elif self.option_type == 'put' and self.strike - spot_price > derivative_value:
            return self.strike - spot_price
        else:
            return None


    def recursion(self, n_iter, spot_price):
        '''
        binomial tree recursion algorithm
        '''

        if n_iter > 1:

            derivative_value = (self.p_tilde*self.recursion(n_iter-1, self.u*spot_price) + self.q_tilde*self.recursion(n_iter-1, self.d*spot_price))
            updated_value = self.on_path_update(spot_price, derivative_value)

            if updated_value != None:
                return updated_value
            else:
                return derivative_value

        else:
            return self.payoff(spot_price)


    def value(self):
        '''
        valuation of the derivative
        '''

        return np.exp(-(self.rate-self.dividend)*self.tau) * self.recursion(self.n_intervals, self.spot_price)





class TrinomialTree(Price):
    '''
    trinomial tree pricing model base class
    '''

    def __init__(
        self,
        init_price,
        maturity,
        rate,
        sigma,
        nper_per_year,
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity)

        self.rate = rate
        self.sigma = sigma
        self.dividend = dividend
        self.nper_per_year = nper_per_year
        self.delta_time = 1 / nper_per_year
        self.n_intervals = int(nper_per_year * self.tau)

        self.u = np.exp(self.sigma*np.sqrt(3*self.delta_time))
        self.d = np.exp(-self.sigma*np.sqrt(3*self.delta_time))

        self.p_u = -np.sqrt(self.delta_time/(12*self.sigma**2))*(self.rate-self.dividend-0.5*self.sigma**2) + (1/6)
        self.p_m = 2/3
        self.p_d = np.sqrt(self.delta_time/(12*self.sigma**2))*(self.rate-self.dividend-0.5*self.sigma**2) + (1/6)


    @abc.abstractmethod
    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        return None


    def on_path_update(self, spot_price, derivative_value):
        '''
        update the derivatives prices on each node
        '''

        return None


    def recursion(self, n_iter, spot_price):
        '''
        trinomial tree recursion algorithm
        '''

        if n_iter > 1:

            derivative_value = np.exp(-(self.rate-self.dividend)*self.delta_time) * (self.p_u*self.recursion(n_iter-1, self.u*spot_price) + self.p_m*self.recursion(n_iter-1, spot_price) + self.p_d*self.recursion(n_iter-1, self.d*spot_price))
            updated_value = self.on_path_update(spot_price, derivative_value)

            if updated_value != None:
                return updated_value
            else:
                return derivative_value

        else:
            return self.payoff(spot_price)


    def value(self):
        '''
        valuation of the derivative
        '''

        return self.recursion(self.n_intervals, self.spot_price)





class EuropeanVanillaTrinomial(TrinomialTree):
    '''
    trinomial tree pricing for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        nper_per_year, 
        strike, 
        option_type='call'):
        '''
        intiaialize the model parameters
        '''
        
        super().__init__(init_price, maturity, rate, sigma, nper_per_year)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type


    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0





class FiniteDifference(Price):
    '''
    finite difference pricing model base class
    '''

    def __init__(
        self,
        init_price,
        maturity,
        rate,
        sigma,
        s_max,
        nper_per_year,
        n_price_intervals,
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity)

        self.rate = rate
        self.sigma = sigma
        self.s_max = s_max
        self.dividend = dividend
        self.nper_per_year = nper_per_year
        self.delta_time = 1 / nper_per_year
        self.n_time_intervals = int(nper_per_year * self.tau)
        self.n_price_intervals = n_price_intervals
        self.delta_s = s_max / n_price_intervals

        self.__compute_matrices()


    def __alpha_j(self, j):
        '''
        compute alpha
        '''

        return (self.delta_time/4) * (self.sigma**2 * j**2 - (self.rate-self.dividend)*j)


    def __beta_j(self, j):
        '''
        compute beta
        '''

        return -(self.delta_time/2) * (self.sigma**2 * j**2 + (self.rate-self.dividend))


    def __gamma_j(self, j):
        '''
        compute gamma
        '''

        return (self.delta_time/4) * (self.sigma**2 * j**2 + (self.rate-self.dividend)*j)


    def __compute_matrices(self):
        '''
        compute model matrices
        '''

        self.m1_matrix = np.zeros((self.n_price_intervals, self.n_price_intervals))
        self.m2_matrix = np.zeros((self.n_price_intervals, self.n_price_intervals))
        
        for j in range(self.n_price_intervals):

            alpha = self.__alpha_j(j+2)
            beta = self.__beta_j(j+1)
            gamma = self.__gamma_j(j+1)

            self.m1_matrix[j, j] = 1 - beta
            self.m2_matrix[j, j] = 1 + beta
            if j+1 < self.n_price_intervals:
                self.m1_matrix[j, j+1] = -gamma
                self.m2_matrix[j, j+1] = gamma
                self.m1_matrix[j+1, j] = -alpha
                self.m2_matrix[j+1, j] = alpha

        self.inv_m1_matrix = la.inv(self.m1_matrix)
            

    @abc.abstractmethod
    def payoff(self, spot_price):
        '''
        define the terminal payoff for a certain derivative
        '''

        return None

    @abc.abstractmethod
    def boundary_value(self, boundary_type, time_i):
        '''
        define the boundary condition for a certain derivative 
        '''

        assert boundary_type in ('up','low'), 'Boundary type must be up or low'

        return None


    def on_path_update(self):
        '''
        update the derivatives prices on each node
        '''

        pass


    def value(self):
        '''
        valuation of the derivative
        '''

        try:
            self.value_vector

        except:

            self.value_vector = np.array([[self.payoff((j+1)*self.delta_s) for j in range(self.n_price_intervals)]]).T

            for i in range(self.n_time_intervals):

                b_vector = np.zeros((self.n_price_intervals, 1))
                b_vector[0] = self.__alpha_j(1) * (self.boundary_value('low', i+1)+self.boundary_value('low', i+2))                           
                b_vector[-1] = self.__alpha_j(self.n_price_intervals-1) * (self.boundary_value('up', i+1)+self.boundary_value('up', i+2))      

                self.value_vector = self.inv_m1_matrix @ (self.m2_matrix @ self.value_vector + b_vector)

                self.on_path_update()

        return self.value_vector[np.argmin(np.abs(self.spot_price-np.array([(j+1)*self.delta_s for j in range(self.n_price_intervals)])))][0]





class EuropeanVanillaFiniteDiff(FiniteDifference):
    '''
    finite difference pricing for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        s_max, 
        nper_per_year, 
        n_price_intervals, 
        strike, 
        option_type='call',
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, rate, sigma, s_max, nper_per_year, n_price_intervals, dividend)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type

    def payoff(self, spot_price):
        '''
        define the terminal payoff of a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0

    def boundary_value(self, boundary_type, time_i):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            if boundary_type == 'up':
                return self.s_max - self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))
            elif boundary_type == 'low':
                return 0
        elif self.option_type == 'put':
            if boundary_type == 'up':
                return 0
            elif boundary_type == 'low':
                return self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))





class EuropeanBarrierFiniteDiff(FiniteDifference):
    '''
    finite difference pricing for european barrier options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        s_max, 
        nper_per_year, 
        n_price_intervals, 
        strike, 
        barrier_up=None, 
        barrier_down=None,
        option_type='call', 
        knock_type='out', 
        direction='up',  
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, rate, sigma, s_max, nper_per_year, n_price_intervals, dividend)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert knock_type.lower() in ('in', 'out'), 'Knock type must be in or out'
        assert direction.lower() in ('up', 'down', 'double'), 'Direction type must be up or down or double'
        if direction == 'up':
            assert barrier_up != None , 'Please input an upward barrier'
        elif direction == 'down':
            assert barrier_down != None , 'Please input a downward barrier'
        elif direction == 'double':
            assert barrier_up != None and barrier_down != None, 'Please input both an upward barrier and a downward barrier'

        self.strike = strike
        self.option_type = option_type
        self.knock_type = knock_type
        self.direction = direction
        self.barrier_up = init_price*barrier_up
        self.barrier_down = init_price*barrier_down


    def payoff(self, spot_price):
        '''
        define the terminal payoff of a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0


    def boundary_value(self, boundary_type, time_i):
        '''
        define the boundary condition for a certain derivative 
        '''

        if self.direction == 'up':
            if self.knock_type == 'out':
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))      
            elif self.knock_type == 'in':   
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return self.s_max - self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0
        if self.direction == 'down':
            if self.knock_type == 'out':
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return self.s_max - self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0
            elif self.knock_type == 'in':   
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time)) 
        if self.direction == 'double':
            if self.knock_type == 'out':
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return 0     
            elif self.knock_type == 'in':   
                if self.option_type == 'call':
                    if boundary_type == 'up':
                        return self.s_max - self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))
                    elif boundary_type == 'low':
                        return 0
                elif self.option_type == 'put':
                    if boundary_type == 'up':
                        return 0
                    elif boundary_type == 'low':
                        return self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time)) 


    def on_path_update(self):
        '''
        update the derivatives prices on each node
        '''

        spot_price_vector = np.array([[(j+1)*self.delta_s for j in range(self.n_price_intervals)]]).T

        if self.direction == 'up':
            if self.knock_type == 'out':
                self.value_vector = np.where(spot_price_vector > self.barrier_up, 0, 1) * self.value_vector
            elif self.knock_type == 'in':
                self.value_vector = np.where(spot_price_vector > self.barrier_up, 1, 0) * self.value_vector
        elif self.direction == 'down':
            if self.knock_type == 'out':
                self.value_vector = np.where(spot_price_vector < self.barrier_down, 0, 1) * self.value_vector
            elif self.knock_type == 'in':
                self.value_vector = np.where(spot_price_vector < self.barrier_down, 1, 0) * self.value_vector
        elif self.direction == 'double':
            if self.knock_type == 'out':
                self.value_vector = np.where(spot_price_vector > self.barrier_up, 0, 1) * np.where(spot_price_vector < self.barrier_down, 0, 1) * self.value_vector
            elif self.knock_type == 'in':
                self.value_vector = (1-(np.where(spot_price_vector > self.barrier_up, 0, 1) * np.where(spot_price_vector < self.barrier_down, 0, 1))) * self.value_vector
                




class AmericanVanillaFiniteDiff(FiniteDifference):
    '''
    finite difference pricing for american vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        s_max, 
        nper_per_year, 
        n_price_intervals, 
        strike, 
        option_type='call', 
        dividend=0):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, rate, sigma, s_max, nper_per_year, n_price_intervals, dividend)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type

    def payoff(self, spot_price):
        '''
        define the terminal payoff of a certain derivative
        '''

        if self.option_type == 'call':
            return spot_price - self.strike if spot_price - self.strike > 0 else 0
        elif self.option_type == 'put':
            return self.strike - spot_price if spot_price - self.strike < 0 else 0


    def boundary_value(self, boundary_type, time_i):
        '''
        define the terminal payoff for a certain derivative
        '''

        if self.option_type == 'call':
            if boundary_type == 'up':
                return self.s_max - self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))
            elif boundary_type == 'low':
                return 0
        elif self.option_type == 'put':
            if boundary_type == 'up':
                return 0
            elif boundary_type == 'low':
                return self.strike*np.exp(-self.rate*(self.tau-time_i*self.delta_time))


    def on_path_update(self):
        '''
        update the derivatives prices on each node
        '''

        spot_price_vector = np.array([[(j+1)*self.delta_s for j in range(self.n_price_intervals)]]).T

        if self.option_type == 'call':
            self.value_vector = np.maximum(spot_price_vector - self.strike, self.value_vector)
        elif self.option_type == 'put':
            self.value_vector = np.maximum(self.strike - spot_price_vector, self.value_vector)

    



class LeastSquaredMonteCarloSim(MonteCarloSim):
    '''
    least squared monte carlo simulation base class
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials,
        nper_per_year, 
        regression_model,
        fetures_degree,
        model, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        try:
            self.model_params['rate']
        except:
            raise ValueError('Longstaff-Schwartz method requires a discounted rate')

        regression_model_list = [
            LinearRegression,
            Ridge,
            Lasso,
            LassoLars,
            ElasticNet,
            KNeighborsRegressor,
            SVR,
            MLPRegressor,
            DecisionTreeRegressor,
            ExtraTreeRegressor,
            RandomForestRegressor,
            AdaBoostRegressor,
            GradientBoostingRegressor,
            BaggingRegressor]
        assert regression_model in regression_model_list, 'Please select an available regression model'
        assert isinstance(fetures_degree, int) and fetures_degree >= 1, 'Number of fetures must be integer and no less than 1'

        self.regression_model = regression_model
        self.fetures_degree = fetures_degree
        self.delta_time = 1 / nper_per_year


    def generate_paths(func):
        '''
        paths generate decorator, the overwritten value method must be decorated by this decorator
        in order to generate paths again when the parameters may be changed
        '''

        def wrapper(self):

            path = self.model(self.n_trials, self.n_intervals, **self.model_params)
            self.price_process = path.generate_paths()

            return func(self)

        return wrapper


    @abc.abstractmethod
    def payoff(self, time_i=None):
        '''
        define the exercise payoff for a certain derivative
        '''

        return None

    @abc.abstractmethod
    def itm_path_index(self, time_i):
        '''
        filter the in the money paths
        '''

        return None


    @generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        self.exercise_matrix = np.where(self.payoff(-1) > 0, 1,0).reshape(self.n_trials, 1)

        for i in range(self.n_intervals-1):

            itm_index = self.itm_path_index(-i-2)
            if len(itm_index) < 1:
                self.exercise_matrix = np.concatenate([np.array([0]*self.n_trials).reshape(len(exercise_vector),1), self.exercise_matrix], axis=1)

            itm_paths = self.price_process[itm_index].reshape(len(itm_index), self.n_intervals)
            dis_cash_flow = np.exp(-self.rate*self.delta_time)*(self.payoff(-i-1)[np.array(itm_index)])           

            reg_X = np.ones((len(itm_index), 1))
            for j in range(self.fetures_degree):
                reg_X = np.concatenate([reg_X, np.array([[x_value**(j+1) for x_value in itm_paths[:,-i-2]]]).T], axis=1)

            regressor = self.regression_model().fit(reg_X, dis_cash_flow)
            reg_result = regressor.predict(reg_X)

            exercise_vector = np.array([0]*self.n_trials) 
            np.put(exercise_vector, itm_index, np.where(self.payoff(-i-2)[np.array(itm_index)] > reg_result, 1,0))
            self.exercise_matrix = np.concatenate([np.array(exercise_vector).reshape(len(exercise_vector),1), self.exercise_matrix], axis=1)

        exercise_index = np.squeeze(np.argmax(self.exercise_matrix, axis=1))
        exercise_indicator = np.max(self.exercise_matrix, axis=1)
        exercise_payoff = self.payoff()[range(self.n_trials), exercise_index] * exercise_indicator

        expire_maturity = ((exercise_index + 1) * exercise_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * exercise_payoff)

        return simulated_price





class AmericanVanillaLSMC(LeastSquaredMonteCarloSim):
    '''
    least squared monte carlo simulation for american vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year,
        strike,
        option_type='put', 
        regression_model=LinearRegression,
        fetures_degree=2,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        super().__init__(init_price, maturity, n_trials, nper_per_year, regression_model, fetures_degree, model, **model_params)

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type

        
    def payoff(self, time_i=None):
        '''
        define the exercise payoff for a certain derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        if time_i != None:
            return relu_func(self.price_process[:,time_i])
        else:
            return relu_func(self.price_process)


    def itm_path_index(self, time_i):
        '''
        filter the in the money paths
        '''

        if self.option_type == 'call':
            return np.argwhere(self.price_process[:,time_i] > self.strike)
        elif self.option_type == 'put':
            return np.argwhere(self.price_process[:,time_i] < self.strike)





class AmericanAsianLSMC(LeastSquaredMonteCarloSim):
    '''
    least squared monte carlo simulation for american asian options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year,
        strike,
        option_type='put',
        ave_type='arith',
        regression_model=LinearRegression,
        fetures_degree=2,
        model=GeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert ave_type.lower() in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        super().__init__(init_price, maturity, n_trials, nper_per_year, regression_model, fetures_degree, model, **model_params)

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type

    
    def payoff(self, time_i=None):
        '''
        define the exercise payoff for a certain derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        if time_i != None:
            if self.ave_type == 'arith':
                return relu_func(np.mean(self.price_process[:,:time_i], axis=1))
            elif self.ave_type == 'geo':
                return relu_func(np.exp(np.mean(np.log(self.price_process[:,:time_i]), axis=1)))
        else:
            if self.ave_type == 'arith':
                return relu_func(np.cumsum(self.price_process, axis=1)/np.array([[i+1 for i in range(self.n_intervals)]]*self.n_trials))
            elif self.ave_type == 'geo':
                return relu_func(np.exp(np.cumsum(np.log(self.price_process), axis=1)/np.array([[i+1 for i in range(self.n_intervals)]]*self.n_trials)))


    def itm_path_index(self, time_i):
        '''
        filter the in the money paths
        '''

        if self.option_type == 'call':
            if self.ave_type == 'arith':
                return np.argwhere(np.mean(self.price_process[:,:time_i], axis=1) > self.strike)
            elif self.ave_type == 'geo':
                return np.argwhere(np.exp(np.mean(np.log(self.price_process[:,:time_i]), axis=1)) > self.strike)
        elif self.option_type == 'put':
            if self.ave_type == 'arith':
                return np.argwhere(np.mean(self.price_process[:,:time_i], axis=1) < self.strike)
            elif self.ave_type == 'geo':
                return np.argwhere(np.exp(np.mean(np.log(self.price_process[:,:time_i]), axis=1)) < self.strike)   





class Quadrature(Price):
    '''
    quadrature base class
    '''

    def __init__(
        self,
        init_price,
        maturity,
        n_intervals,
        integral_type,
        model_type,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity)

        self.n_intervals = n_intervals
        self.integral_type = integral_type
        self.model_type = model_type
        self.model_params = model_params
        self.weight = 6/self.n_intervals

        self.pdf_type = None

        model_dict = {
            'gbm':['mu', 'sigma'],
            'bachelier':['mu', 'sigma']}
        assert model_type in model_dict.keys(), 'Please select an avilable underlying price distribution'
        assert integral_type in ('left', 'middle', 'trapezoidal'), 'Please select an avilable integration type' # 'gaussian'

        try:
            if self.model_type == 'gbm':
                self.mu = self.model_params['mu']
                self.sigma = self.model_params['sigma']
                self.pdf_type = 'normal'
            elif self.model_type == 'bachelier':
                self.mu = self.model_params['mu']
                self.sigma = self.model_params['sigma']   
                self.pdf_type = 'normal'
        except:
            raise NameError('Distribution parameters input are not correct')
        
        
    @abc.abstractmethod
    def payoff(self, s):
        '''
        define the exercise payoff for a certain derivative
        '''

        return None


    def pdf(self, y):
        '''
        return the pdf function value for a certain distribution
        '''

        if self.pdf_type == 'normal':
            pdf_value =  (1/np.sqrt(2*np.pi))*np.exp(-0.5*y**2)
        
        return pdf_value


    def model_price(self, x):
        '''
        return the analyatical spot price for a certain model
        '''

        if self.model_type == 'gbm':
            current_price = self.spot_price*np.exp((self.mu-0.5*self.sigma**2)*self.tau+self.sigma*np.sqrt(self.tau)*x)
        elif self.model_type == 'bachelier':
            current_price = self.spot_price*np.exp(-self.mu*self.tau)+self.sigma*np.sqrt((np.exp(2*self.mu*self.tau)-1)/(2*self.mu))*x

        return current_price


    def value(self):
        '''
        valuation of the derivative
        '''

        quadrature_sum = 0

        for i in range(self.n_intervals):

            if self.integral_type == 'middle':
                quadrature_sum += self.weight*self.payoff(self.model_price(-3+self.weight*(i+0.5)))*self.pdf(-3+self.weight*(i+0.5))
            elif self.integral_type == 'left':
                quadrature_sum += self.weight*self.payoff(self.model_price(-3+self.weight*i))*self.pdf(-3+self.weight*i)
            elif self.integral_type == 'trapezoidal':
                quadrature_sum += 0.5*(self.weight*self.payoff(self.model_price(-3+self.weight*i))*self.pdf(-3+self.weight*i) + self.weight*self.payoff(self.model_price(-3+self.weight*(i+1)))*self.pdf(-3+self.weight*(i+1)))

        return quadrature_sum





class EuropeanVanillaQuadrature(Quadrature):
    '''
    quadrature pricing for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_intervals, 
        strike, 
        option_type='call', 
        integral_type='middle', 
        model_type='gbm', 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_intervals, integral_type, model_type, **model_params)

        self.strike = strike
        self.option_type = option_type


    def payoff(self, s):
        '''
        define the exercise payoff for a certain derivative
        '''

        if self.option_type == 'call':
            if s >= self.strike:
                return s - self.strike
            else:
                return 0
        elif self.option_type == 'put':
            if s <= self.strike:
                return self.strike - s
            else:
                return 0





