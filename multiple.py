'''
File Name : multiple.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu


'''

import abc
import numpy as np
import numpy.linalg as la
from paths import *
from pricing import * 
import warnings
warnings.filterwarnings("ignore")




class MultiMonteCarloSim(MonteCarloSim):
    '''
    Multiple Monte Carlo simulation abstract base class which should only be inherited by subclasses
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

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        model_types_list = [
            MultiCEV, 
            MultiGeoBrownianMotion]
        assert model in model_types_list, 'Selcet an available model type'

        self.n_assets = len(init_price)



    @abc.abstractmethod
    @MonteCarloSim.generate_paths
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


    def std(self, index):
        '''
        compute standard variation of the ith sample paths
        '''

        assert index <= self.n_assets, 'The index of ith asset is greater than the number of assets'

        try:
            return np.std(self.price_process[index-1,:,-1])
        except:
            raise ValueError('There is no payoff which is computed')





class EuropeanVanillaSpreadMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european vanilla spread options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert self.n_assets == 2, 'The number of assets must be 2'
        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        self.payoff_vector = relu_func(self.price_process[0,:,-1]-self.price_process[1,:,-1])
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price




class EuropeanVanillaExchangeMonteCarloSim(EuropeanVanillaSpreadMonteCarloSim):
    '''
    monte carlo simulation for european vanilla exchange options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, 0, 'call', model, **model_params)





class EuropeanVanillaBasketMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european vanilla basket options
    '''

    def __init__(
        self, 
        init_price, 
        weight,
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert len(weight) == self.n_assets, 'The number of weights must be euqal to the number of the assets'
        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.weight = np.array(weight/np.sum(weight)).reshape(self.n_assets,1)
        self.strike = strike
        self.option_type = option_type


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        self.payoff_vector = relu_func(self.weight.T @ self.price_process[:,:,-1])
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class EuropeanVanillaRainbowMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european vanilla rainbow options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        payoff_type='max',
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type
        self.payoff_type = payoff_type


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)

        if self.payoff_type == 'max':
            self.payoff_vector = relu_func(np.max(self.price_process[:,:,-1], axis=0))
        elif self.payoff_type == 'min':
            self.payoff_vector = relu_func(np.min(self.price_process[:,:,-1], axis=0))

        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class EuropeanBarrierSpreadMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european barrier spread options
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
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert self.n_assets == 2, 'The number of assets must be 2'
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
        self.barrier_up = (init_price[0]-init_price[1])*barrier_up
        self.barrier_down = init_price*barrier_down


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        spread_process = self.price_process[0,:,:]-self.price_process[1,:,:]
        indicator_matrix = np.zeros((self.n_trials,1))

        if self.direction == 'up':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.max(spread_process,axis=1) < self.barrier_up, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = np.array(np.max(spread_process,axis=1) >= self.barrier_up, dtype=int)
        elif self.direction == 'down':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.min(spread_process,axis=1) > self.barrier_down, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = np.array(np.min(spread_process,axis=1) <= self.barrier_down, dtype=int)
        elif self.direction == 'double':
            if self.knock_type == 'out':
                indicator_matrix = np.array(np.max(spread_process,axis=1) < self.barrier_up, dtype=int) * np.array(np.min(spread_process,axis=1) > self.barrier_down, dtype=int)
            elif self.knock_type == 'in':
                indicator_matrix = 1-((1-np.array(np.max(spread_process,axis=1) >= self.barrier_up, dtype=int)) * (1-np.array(np.min(spread_process,axis=1) <= self.barrier_down, dtype=int)))

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1)  

        self.payoff_vector = relu_func(spread_process[:,-1])*indicator_matrix
        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price




class EuropeanAsianBasketMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european asian basket options
    '''

    def __init__(
        self, 
        init_price, 
        weight,
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        ave_type='arith',
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert len(weight) == self.n_assets, 'The number of weights must be euqal to the number of the assets'
        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert ave_type.lower() in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        self.weight = np.array(weight/np.sum(weight)).reshape(self.n_assets,1)
        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:x-self.strike if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:self.strike-x if self.strike-x >= 0 else 0,1,1) 

        if self.ave_type == 'arith':
            self.payoff_vector = relu_func(np.mean(np.tensordot(self.weight.T, self.price_process, axes=1), axis=1))
        elif self.ave_type == 'geo':
            self.payoff_vector = relu_func(np.exp(np.mean(np.log(np.tensordot(self.weight.T, self.price_process, axes=1)), axis=1)))

        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price




class EuropeanBinaryRainbowMonteCarloSim(MultiMonteCarloSim):
    '''
    monte carlo simulation for european binary rainbow options
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials, 
        nper_per_year, 
        strike, 
        option_type='call',
        payoff_type='max',
        model=MultiGeoBrownianMotion, 
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.strike = strike
        self.option_type = option_type
        self.payoff_type = payoff_type


    @MultiMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        if self.option_type == 'call':
            relu_func = np.frompyfunc(lambda x:1 if x-self.strike >= 0 else 0,1,1)
        elif self.option_type == 'put':
            relu_func = np.frompyfunc(lambda x:1 if self.strike-x >= 0 else 0,1,1)

        if self.payoff_type == 'max':
            self.payoff_vector = relu_func(np.max(self.price_process[:,:,-1], axis=0))
        elif self.payoff_type == 'min':
            self.payoff_vector = relu_func(np.min(self.price_process[:,:,-1], axis=0))

        simulated_price = np.exp(-self.rate*self.tau) * np.mean(self.payoff_vector)

        return simulated_price





class MultiLeastSquaredMonteCarloSim(LeastSquaredMonteCarloSim,MonteCarloSim):
    '''
    multiple least squared monte carlo simulation base class
    '''

    def __init__(
        self, 
        init_price, 
        maturity, 
        n_trials,
        nper_per_year, 
        regression_model,
        features_degree,
        model,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, regression_model, features_degree, model, **model_params)

        
        model_types_list = [
            MultiCEV, 
            MultiGeoBrownianMotion]
        assert model in model_types_list, 'Selcet an available model type'

        self.n_assets = len(init_price)


    @abc.abstractmethod
    def process_to_2d(self):
        '''
        reduce the assets dimension on process tensor by the payoff structure
        '''

        pass



    def std(self, index):
        '''
        compute standard variation of the ith sample paths
        '''

        return MultiMonteCarloSim.std(index)


    @LeastSquaredMonteCarloSim.generate_paths
    def value(self):
        '''
        valuation of the derivative
        '''

        self.process_to_2d()

        self.exercise_matrix = np.where(self.payoff(-1) > 0, 1,0).reshape(self.n_trials, 1)

        for i in range(self.n_intervals-1):

            itm_index = self.itm_path_index(-i-2)
            if len(itm_index) < 1:
                self.exercise_matrix = np.concatenate([np.array([0]*self.n_trials).reshape(self.n_trials,1), self.exercise_matrix], axis=1)
                continue

            itm_paths = self.price_process[itm_index].reshape(len(itm_index), self.n_intervals)
            dis_cash_flow = np.exp(-self.rate*self.delta_time)*(self.payoff(-i-1)[np.array(itm_index)])           

            reg_X = np.ones((len(itm_index), 1))
            for j in range(self.features_degree):
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





class AmericanVanillaBasketLSMC(MultiLeastSquaredMonteCarloSim):
    '''
    least squared monte carlo simulation for american vanilla basket options
    '''

    def __init__(
        self, 
        init_price, 
        weight,
        maturity, 
        n_trials, 
        nper_per_year,
        strike,
        option_type='put', 
        regression_model=LinearRegression,
        features_degree=2,
        model=MultiGeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, regression_model, features_degree, model, **model_params)

        assert len(weight) == self.n_assets, 'The number of weights must be euqal to the number of the assets'
        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'

        self.rate = model_params['rate']
        self.weight = np.array(weight/np.sum(weight)).reshape(self.n_assets,1)
        self.strike = strike
        self.option_type = option_type


    def process_to_2d(self):
        '''
        reduce the assets dimension on process tensor by the payoff structure
        '''

        self.price_process = np.tensordot(self.weight.T, self.price_process, axes=1).reshape(self.n_trials, self.n_intervals)

        
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




class AmericanAsianRainbowLSMC(MultiLeastSquaredMonteCarloSim):
    '''
    least squared monte carlo simulation for american asian rainbow options
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
        payoff_type='max',
        regression_model=LinearRegression,
        features_degree=2,
        model=MultiGeoBrownianMotion,
        **model_params):
        '''
        intiaialize the structure parameters
        '''

        super().__init__(init_price, maturity, n_trials, nper_per_year, regression_model, features_degree, model, **model_params)

        assert option_type.lower() in ('call', 'put'), 'The option type must be call or put'
        assert ave_type.lower() in ('arith', 'geo'), 'The average type must be arithmetic or geometric'

        self.rate = model_params['rate']
        self.strike = strike
        self.option_type = option_type
        self.ave_type = ave_type
        self.payoff_type = payoff_type


    def process_to_2d(self):
        '''
        reduce the assets dimension on process tensor by the payoff structure
        '''

        if self.payoff_type == 'max':
            self.price_process = np.max(self.price_process, axis=0)
        elif self.payoff_type == 'min':
            self.price_process = np.min(self.price_process, axis=0)


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



