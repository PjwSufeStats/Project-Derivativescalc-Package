'''
File Name : pricing.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This file covers the fundamental pricing approaches for different derivatives structures,
The monte carlo simulation approach, binominal trees approach and finite difference approach has been
introduced into pricing. The monte carlo approach allows user to select different underlying dynamics
in paths.py to generate paths. Futhermore, this file also contains the pricing approaches for some popupar 
OTC derivatives such as snow ball product.
'''

import abc
from scipy.stats import norm
from paths import * 


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

    def delta_func(pos_or_neg, maturity, u):

        if pos_or_neg:
            return (1/(sigma*np.sqrt(maturity))) * (np.log(u) + (rate+0.5*sigma**2)*maturity)
        else:
            return (1/(sigma*np.sqrt(maturity))) * (np.log(u) + (rate-0.5*sigma**2)*maturity)

    I1 = norm.cdf(delta_func(True, maturity, spot_price/strike_price)) - norm.cdf(delta_func(True, maturity, spot_price/barrier))
    I2 = np.exp(rate*maturity) * (norm.cdf(delta_func(False, maturity, spot_price/strike_price)) - norm.cdf(delta_func(False, maturity, spot_price/barrier)))
    I3 = (spot_price/barrier)**(-(2*rate/sigma**2)-1) * (norm.cdf(delta_func(True, maturity, barrier**2/(strike_price*spot_price))) - norm.cdf(delta_func(True, maturity, barrier/spot_price)))
    I4 = np.exp(-rate*maturity) * (spot_price/barrier)**(-(2*rate/sigma**2)-1) * (norm.cdf(delta_func(False, maturity, barrier**2/(strike_price*spot_price))) - norm.cdf(delta_func(False, maturity, barrier/spot_price)))

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
            Heston]
        assert model in model_types_list, 'Selcet an available model type'

        self.n_trials = n_trials
        self.nper_per_year = nper_per_year
        self.n_intervals = nper_per_year * self.tau
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
            self.n_intervals = self.nper_per_year * self.tau
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

        if 'sigma' in self.model_params.keys():
            self.reset_param('sigma', new_sigma)
        elif 'initialized_volatility' in self.model_params.keys():
            self.reset_param('initialized_volatility', new_sigma)
        else:
            raise NameError('This model does not have a volatility related parameter')

        return self.value()


    def price_update_rate(self, new_rate=None):
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

    def __init__(self, 
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

        assert option_type in ('call', 'put'), 'The option type must be call or put'

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
        use_gpu=True, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        assert option_type in ('call', 'put'), 'The option type must be call or put'
        assert knock_type in ('in', 'out'), 'Knock type must be in or out'
        assert direction in ('up', 'down', 'double'), 'Direction type must be up or down or double'
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
        self.barrier_up = init_price* barrier_up
        self.barrier_down = init_price* barrier_down


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
                indicator_matrix = np.array(np.max(self.price_process,axis=1) >= self.barrier_up, dtype=int) * np.array(np.min(self.price_process,axis=1) <= self.barrier_down, dtype=int)

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
        use_gpu=True, 
        **model_params):
        '''
        intiaialize the model parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        assert option_type in ('call', 'put'), 'The option type must be call or put'

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
        observe_days=None,
        model=GeoBrownianMotion, 
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        assert option_type in ('call', 'put'), 'The option type must be call or put'
        assert ave_type in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type
        self.observe_days = observe_days


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
        use_gpu=True, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        assert option_type in ('call', 'put'), 'The option type must be call or put'

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
        nominal_principal,
        fixed_yield,
        ki_level,
        ko_level,
        low_strike=0,
        high_strike=1,
        step_down=0,
        ki_fre='day',
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        fre_dict = {'day':1, 'month':30, 'year':360, None:None}
        assert ki_fre in fre_dict.keys() and ko_fre in fre_dict.keys() and ko_fre in sd_fre.keys(), 'The input knock in or knock out frequency is not valid'

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        self.rate = model_params['rate']
        self.nominal_principal = nominal_principal
        self.fixed_yield = fixed_yield
        self.ki_barrier = init_price * ki_level
        self.ko_barrier = init_price * ko_level
        self.low_strike = init_price * low_strike
        self.high_strike = init_price * high_strike
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
        ko_payoff = (np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator * (self.fixed_yield/self.nper_per_year)
        nki_nko_payoff = self.fixed_yield*self.tau * (1 - ki_indicator) * (1 - ko_indicator)
        self.payoff_vector = self.nominal_principal * (ko_payoff + ki_payoff*(1 - ko_indicator) + nki_nko_payoff)

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
        nominal_principal,
        fixed_yield,
        ki_level,
        ko_level,
        low_strike=0,
        high_strike=1,
        step_down=0,
        ki_fre='day',
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, nominal_principal, fixed_yield, ki_level, ko_level, low_strike, 
                        high_strike, step_down, ki_fre, ko_fre, sd_fre, skip_ko_observe, model, use_gpu, **model_params)


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

        ko_payoff = np.sum(yield_payoff[:,:(np.argmax(ko_indicator_matrix, axis=1)+1)*ko_indicator], axis=1) * ((self.fixed_yield*self.ko_interval)/self.nper_per_year)

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if self.spot_price-x >= 0 else 0,1,1)    
        ki_payoff = (relu_func(self.price_process[:,-1]) + (np.sum(yield_payoff, axis=1) * ((self.fixed_yield*self.ko_interval)/self.nper_per_year))) * ki_indicator

        nki_nko_payoff = self.fixed_yield*self.tau * (1 - ki_indicator) * (1 - ko_indicator)
        self.payoff_vector = self.nominal_principal * (ko_payoff + ki_payoff*(1 - ko_indicator) + nki_nko_payoff)

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
        nominal_principal,
        fixed_yield,
        ki_level,
        ko_level,
        low_strike=0,
        high_strike=1,
        step_down=0,
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, nominal_principal, fixed_yield, ki_level, ko_level, low_strike, 
                        high_strike, step_down, None, ko_fre, sd_fre, skip_ko_observe, model, use_gpu, **model_params)


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''
        
        ko_barrier_matrix = np.array([[(lambda x:self.ko_barrier*(1 - self.step_down*(x//self.sd_interval)))(i) for i in range(self.n_intervals)]]*self.n_trials)
        ko_indicator_matrix = (np.where(self.price_process >= ko_barrier_matrix, 1, 0) * 
                               np.array([[(lambda x:1 if (x+1)%self.ko_interval==0 and float(x+1)/self.ko_interval > self.skip_ko_observe else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ko_indicator = np.max(ko_indicator_matrix, axis=1)

        ko_payoff = (np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator * (self.fixed_yield/self.nper_per_year)

        relu_func = np.frompyfunc(lambda x:(x/self.ki_barrier)-1 if self.ki_barrier-x >= 0 else 0,1,1) 
        no_ki_payoff = (relu_func(self.price_process[:,-1]) + (self.fixed_yield*self.tau))

        self.payoff_vector = self.nominal_principal * (ko_payoff*ko_indicator + no_ki_payoff*(1-ko_indicator))

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
        nominal_principal,
        ki_level,
        low_strike=0,
        high_strike=1,
        ki_fre='day',
        model=GeoBrownianMotion,
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        fre_dict = {'day':1, 'month':30, 'year':360}

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, use_gpu, **model_params)

        self.rate = model_params['rate']
        self.nominal_principal = nominal_principal
        self.ki_barrier = init_price * ki_level
        self.low_strike = init_price * low_strike
        self.high_strike = init_price * high_strike
        self.ki_interval = fre_dict[ki_fre]


    @MonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        ki_indicator_matrix = (np.where(self.price_process < self.ki_barrier, 1, 0) * np.array([[(lambda x:1 if (x+1)%self.ki_interval==0 else 0)(i) for i in range(self.n_intervals)]]*self.n_trials))
        ki_indicator = np.max(ki_indicator_matrix, axis=1)
        
        ki_payoff_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 ,1,1)    
        ki_payoff = ki_payoff_func(self.price_process[:,-1]) * ki_indicator

        relu_func = np.frompyfunc(lambda x:(x/self.spot_price)-1 if self.spot_price-x >= 0 else 0,1,1)
        no_ki_payoff = relu_func(self.price_process[:,-1]) * ki_indicator

        self.payoff_vector = self.nominal_principal * (no_ki_payoff*(1-ki_indicator) + ki_payoff*ki_indicator)
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
        nominal_principal,
        fixed_yield,
        ki_level,
        ko_level,
        participate,
        low_strike=0,
        high_strike=1,
        step_down=0,
        ko_fre='month',
        sd_fre='month',
        skip_ko_observe=0,
        model=GeoBrownianMotion,
        use_gpu=True,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        assert ko_level > 1, 'The knock level must be greater than 100%'
        assert ki_level < 1, 'The knock level must be less than 100%'

        super().__init__(init_price, maturity, n_trials, nper_per_year, nominal_principal, fixed_yield, ki_level, ko_level, low_strike, 
                        high_strike, step_down, None, ko_fre, sd_fre, skip_ko_observe, model, use_gpu, **model_params)

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

        ko_payoff = self.nominal_principal * (self.ko_barrier/self.spot_price - 1)

        relu_func = np.frompyfunc(lambda x:self.participate*((x/self.spot_price)-1) if x-self.spot_price >= 0 else (self.ki_barrier/self.spot_price)-1 if x-self.ki_barrier <= 0 else (x/self.spot_price)-1 ,1,1)
        no_ko_payoff = relu_func(self.price_process[:,-1])

        self.payoff_vector = self.nominal_principal * (ko_payoff*ko_indicator + no_ko_payoff*(1-ko_indicator))

        expire_maturity = ((np.argmax(ko_indicator_matrix, axis=1) + 1) * ko_indicator) / self.nper_per_year
        discount_func = np.frompyfunc(lambda x:np.exp(-self.rate*x),1,1)

        simulated_price = np.mean(discount_func(expire_maturity) * self.payoff_vector)

        return simulated_price









class BinominalTree(Price):
    '''
    binominal tree pricing model base class
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
        self.n_intervals = nper_per_year * self.tau

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

    def path_judgement(self, spot_price):
        '''
        judge the termination conditions in advance
        '''

        return None

    def recursion(self, n_iter, spot_price):
        '''
        binominal tree recursion algorithm
        '''

        path_return = self.path_judgement(spot_price)
        if path_return != None:
            return path_return

        if n_iter > 1:
            return np.exp(-(self.rate-self.dividend)*self.delta_time) * (self.p_tilde*self.recursion(n_iter-1, self.u*spot_price) + self.q_tilde*self.recursion(n_iter-1, self.d*spot_price))
        else:
            return self.payoff(spot_price)

    def value(self):
        '''
        valuation of the derivative
        '''

        return self.recursion(self.n_intervals, self.spot_price)





class EuropeanVanillaBinominal(BinominalTree):
    '''
    binominal tree pricing for european vanilla options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        rate, 
        sigma, 
        nper_per_year, 
        strike, 
        option_type):
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





class EuropeanAsianBinominal(BinominalTree):
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
        recursion algorithm for binominal tree pricing model
        '''

        path_return = self.path_judgement(spot_price)
        if path_return != None:
            return path_return

        if self.ave_type == 'arith':
            on_path_sum += spot_price
        elif self.ave_type == 'geo':
            on_path_sum *= spot_price
        
        if n_iter > 1:
            return np.exp(-(self.rate-self.dividend)*self.delta_time) * (self.p_tilde*self.recursion(n_iter-1, self.u*spot_price, on_path_sum) + self.q_tilde*self.recursion(n_iter-1, self.d*spot_price, on_path_sum))
        else:
            return self.payoff(on_path_sum)




class AmericanVanillaBinominal(BinominalTree):
    '''
    binominal tree pricing for american vanilla options
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

    def path_judgement(self, spot_price):
        '''
        judge the termination conditions in advance
        '''

        if self.option_type == 'call' and spot_price - self.strike > 0:
            return spot_price - self.strike
        elif self.option_type == 'put' and spot_price - self.strike < 0:
            return self.strike - spot_price
        else:
            return None




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
        self.n_time_intervals = nper_per_year * self.tau
        self.n_price_intervals = n_price_intervals
        self.delta_s = s_max / n_price_intervals

        self.__compute_matrices()


    def __alpha_j(self, j):
        '''
        compute alpha
        '''

        return (self.delta_time/4) * (self.sigma**2 * j**2 - self.rate*j)


    def __beta_j(self, j):
        '''
        compute beta
        '''

        return -(self.delta_time/2) * (self.sigma**2 * j**2 + self.rate)


    def __gamma_j(self, j):
        '''
        compute gamma
        '''

        return (self.delta_time/4) * (self.sigma**2 * j**2 + self.rate*j)


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

        return None

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

                self.value_vector = la.inv(self.m1_matrix) @ (self.m2_matrix @ self.value_vector + b_vector)

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

    def boundary_value(self, boundary_type, _=None):
        '''
        define the terminal payoff for a certain derivative
        '''

        if option_type == 'call':
            if boundary_type == 'up':
                return self.s_max - self.strike*np.exp(-self.rate*self.tau)
            elif boundary_type == 'low':
                return 0
        elif option_type == 'put':
            if boundary_type == 'up':
                return 0
            elif boundary_type == 'low':
                return self.strike*np.exp(-self.rate*self.tau)


