'''
File Name : paths.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This file covers the implementation of some basic derivatives
pricing models used for generating simulated sample paths which can be applied
in monte carlo modules in pricing.py file to price different derivatives via
monte carlo simulation approach.
'''

import abc
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")



class Path(metaclass=abc.ABCMeta):
    '''
    path generation base class
    '''

    def __init__(self, n_trials, n_intervals, maturity):
        '''
        intiaialize the model parameters
        '''

        self.n_trials = int(n_trials)
        self.n_intervals = int(n_intervals)
        self.delta_time = maturity / self.n_intervals


    @abc.abstractmethod
    def generate_paths(self):
        '''
        define the generate_paths method previously
        '''

        print('Base class Path has no concrete implementation of .generate_paths(), this method must be implemented in inherited class')

        self.price_paths = None

        return self.price_paths


    def plot_paths(self):
        '''
        plot the simulated paths
        '''

        self.generate_paths()

        plt.figure(figsize=(20,10))

        for i in range(self.price_paths.shape[0]):
            plt.plot(self.price_paths[i,:])

        plt.title('Sample Paths Plot')
            
        plt.show()


    def plot_hist(self):
        '''
        plot the simulated paths histogram
        '''

        self.generate_paths()

        plt.figure(figsize=(20,10))

        plt.hist(self.price_paths[:,-1], bins=100)

        plt.title('Sample Paths Histogram')
            
        plt.show()




class CEV(Path):
    '''
    generate simulated sample paths based on CEV motion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''
        
        try:
            self.spot_price = model_params['spot_price']
            self.rate = model_params['rate']
            self.sigma = model_params['sigma']
            self.tau = model_params['tau']
            self.beta = model_params['beta']
        except:
            raise NameError('Parameters input are wrong')

        super().__init__(n_trials, n_intervals, self.tau)

        
    def generate_paths(self):
        '''
        generate N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))

        for _ in range(self.n_intervals):
            norm_delta = np.random.randn(self.n_trials, 1)
            price_vector = price_vector + self.rate*price_vector*self.delta_time + (price_vector**self.beta) * self.sigma*norm_delta*np.sqrt(self.delta_time)
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths



class Bachelier(CEV):
    '''
    generate simulated sample paths based on bachelier motion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        model_params['beta'] = 0
        super().__init__(n_trials, n_intervals, **model_params)



class GeoBrownianMotion(CEV):
    '''
    generate simulated sample paths based on geometric brownian motion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        model_params['beta'] = 1
        super().__init__(n_trials, n_intervals, **model_params)



class OU(Path):
    '''
    generate simulated sample paths based on OU model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.kappa = model_params['kappa']
            self.theta = model_params['theta']
            self.sigma = model_params['sigma']
            self.tau = model_params['tau']
        except:
            raise NameError('Parameters input are wrong')

        super().__init__(n_trials, n_intervals, self.tau)


    def generate_paths(self):
        '''
        generate N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))

        for _ in range(self.n_intervals):
            norm_delta = np.random.randn(self.n_trials, 1)
            price_vector = price_vector + self.kappa*(self.theta-price_vector)*self.delta_time + self.sigma*norm_delta*np.sqrt(self.delta_time)
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths


class CIR(Path):
    '''
    generate simulated sample paths based on CIR model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.kappa = model_params['kappa']
            self.theta = model_params['theta']
            self.sigma = model_params['sigma']
            self.tau = model_params['tau']
        except:
            raise NameError('Parameters input are wrong')

        super().__init__(n_trials, n_intervals, self.tau)


    def generate_paths(self):
        '''
        generate N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))

        for _ in range(self.n_intervals):
            norm_delta = np.random.randn(self.n_trials, 1)
            price_vector = price_vector + self.kappa*(self.theta-price_vector)*self.delta_time + self.sigma*np.sqrt(price_vector)**norm_delta*np.sqrt(self.delta_time)
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths


class SABR(Path):
    '''
    generate simulated sample paths based on SABR model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.alpha = model_params['alpha']
            self.beta = model_params['beta']
            self.rho = model_params['rho']
            self.init_vol = model_params['initialized_volatility']
            self.tau = model_params['tau']

        except:
            raise NameError('Parameters input are wrong')

        super().__init__(n_trials, n_intervals, self.tau)


    def generate_paths(self):
        '''
        generate N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))
        volatility_vector = self.init_vol * np.ones((self.n_trials,1))

        corr_matrix = la.cholesky(np.array([[1,self.rho],[self.rho,1]]))

        for _ in range(self.n_intervals):

            norm_delta = np.random.randn(self.n_trials, 2) @ corr_matrix
            norm_delta1 = norm_delta[:,0].reshape(self.n_trials,1)
            norm_delta2 = norm_delta[:,1].reshape(self.n_trials,1)

            volatility_vector = volatility_vector + self.alpha*volatility_vector*np.sqrt(self.delta_time)*norm_delta1
            price_vector = price_vector + volatility_vector*(price_vector**self.beta)*np.sqrt(self.delta_time)*norm_delta2
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths


class Heston(Path):
    '''
    generate simulated sample paths based on Heston model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.rate = model_params['rate']
            self.tau = model_params['tau']
            self.init_vol = model_params['initialized_volatility']
            self.rho = model_params['rho']
            self.kappa = model_params['xi']
            self.xi = model_params['kappa']
            self.theta = model_params['theta']
        except:
            raise NameError('Parameters input are wrong')

        super().__init__(n_trials, n_intervals, self.tau)


    def generate_paths(self):
        '''
        generate N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))
        volatility_vector = self.init_vol * np.ones((self.n_trials,1))

        corr_matrix = la.cholesky(np.array([[1,self.rho],[self.rho,1]]))

        for _ in range(self.n_intervals):

            norm_delta = np.random.randn(self.n_trials, 2) @ corr_matrix
            norm_delta1 = norm_delta[:,0].reshape(self.n_trials,1)
            norm_delta2 = norm_delta[:,1].reshape(self.n_trials,1)

            volatility_vector = volatility_vector + self.kappa*(self.theta-np.maximum(volatility_vector,0))*self.delta_time + self.xi*np.sqrt(np.maximum(volatility_vector,0))*np.sqrt(self.delta_time)*norm_delta1
            price_vector = price_vector + self.rate*price_vector*self.delta_time + np.sqrt(np.maximum(volatility_vector,0))*price_vector*np.sqrt(self.delta_time)*norm_delta2

            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths



class Merton(Path):
    '''
    generate simulated sample paths based on Merton's Jump Diffusion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.rate = model_params['rate']
            self.sigma = model_params['sigma']
            self.lambda_ = model_params['lambda']
            self.mu = model_params['mu']
            self.delta = model_params['delta']
            self.tau = model_params['tau']
        except:
            raise ValueError('Parameters input are not correct')

        super().__init__(n_trials, n_intervals, self.tau)

        self.kj = np.exp(self.mu + 0.5*self.delta**2)


    def generate_paths(self):
        '''
        return N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))

        for _ in range(self.n_intervals):
            norm_delta1 = np.random.randn(self.n_trials, 1)
            norm_delta2 = np.random.randn(self.n_trials, 1)
            poisson_delta = np.random.poisson(self.lambda_*self.delta_time, self.n_trials).reshape(self.n_trials, 1)
            jump_t = np.exp(self.mu + self.delta*norm_delta2)

            price_vector = (price_vector + price_vector*(self.rate-self.lambda_*self.kj)*self.delta_time + price_vector*self.sigma*norm_delta1*np.sqrt(self.delta_time) + 
                            (jump_t-1)*price_vector*poisson_delta)

            price_vector = np.maximum(price_vector, 0)
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths



class Bates(Path):
    '''
    generate simulated sample paths based on Bates' Stochastic Volatility Jump model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            self.spot_price = model_params['spot_price']
            self.rate = model_params['rate']
            self.lambda_ = model_params['lambda']
            self.mu = model_params['mu']
            self.delta = model_params['delta']
            self.init_vol = model_params['initialized_volatility']
            self.rho = model_params['rho']
            self.kappa = model_params['xi']
            self.xi = model_params['kappa']
            self.theta = model_params['theta']
            self.tau = model_params['tau']
        except:
            raise ValueError('Parameters input are not correct')

        super().__init__(n_trials, n_intervals, self.tau)

        self.kj = np.exp(self.mu + 0.5*self.delta**2)


    def generate_paths(self):
        '''
        return N * M array, N is the number of paths, M is the number of sampling intervals 
        '''

        self.price_paths = price_vector = self.spot_price * np.ones((self.n_trials, 1))
        volatility_vector = self.init_vol * np.ones((self.n_trials,1))

        corr_matrix = la.cholesky(np.array([[1,self.rho],[self.rho,1]]))

        for _ in range(self.n_intervals):

            norm_delta = np.random.randn(self.n_trials, 2) @ corr_matrix
            norm_delta1 = norm_delta[:,0].reshape(self.n_trials,1)
            norm_delta2 = norm_delta[:,1].reshape(self.n_trials,1)
            norm_delta3 = np.random.randn(self.n_trials, 1)
            poisson_delta = np.random.poisson(self.lambda_*self.delta_time, self.n_trials).reshape(self.n_trials, 1)
            jump_t = np.exp(self.mu + self.delta*norm_delta3)

            volatility_vector = volatility_vector + self.kappa*(self.theta-np.maximum(volatility_vector,0))*self.delta_time + self.xi*np.sqrt(np.maximum(volatility_vector,0))*np.sqrt(self.delta_time)*norm_delta1
            price_vector = (price_vector + price_vector*(self.rate-self.lambda_*self.kj)*self.delta_time + np.sqrt(np.maximum(volatility_vector,0))*price_vector*norm_delta2*np.sqrt(self.delta_time) + 
                            (jump_t-1)*price_vector*poisson_delta)

            price_vector = np.maximum(price_vector, 0)
            self.price_paths = np.concatenate([self.price_paths, price_vector], axis=1)

        self.price_paths = np.delete(self.price_paths, 0, axis=1)

        return self.price_paths





class MultiPath(Path):
    '''
    multiple path generation base class
    '''

    def __init__(self, n_trials, n_intervals, maturity):
        '''
        intiaialize the model parameters
        '''

        self.n_trials = int(n_trials)
        self.n_intervals = int(n_intervals)
        self.delta_time = maturity / self.n_intervals


    def plot_paths(self, index):
        '''
        plot the simulated paths for the ith asset
        '''

        assert index <= self.n_assets, 'The index of ith asset is greater than the number of assets'

        self.generate_paths()

        plt.figure(figsize=(20,10))

        for i in range(self.price_paths.shape[1]):
            plt.plot(self.price_paths[index-1,i,:])

        plt.title('Sample Paths Plot for the {}th Spot Asset'.format(index))
            
        plt.show()

    def plot_hist(self, index):
        '''
        plot the simulated paths histogram for the ith asset
        '''

        assert index <= self.n_assets, 'The index of ith asset is greater than the number of assets'

        self.generate_paths()

        plt.figure(figsize=(20,10))

        plt.hist(self.price_paths[index-1,:,-1], bins=100)

        plt.title('Sample Paths Histogram for the {}th Spot Asset'.format(index))
            
        plt.show()





class MultiCEV(MultiPath):
    '''
    generate multiple simulated sample paths based on CEV motion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''
        
        try:
            self.rate = model_params['rate']  
            self.tau = model_params['tau']
            self.spot_price = np.array(model_params['spot_price'])
            self.sigma = np.array(model_params['sigma'])
            self.beta = np.array(model_params['beta'])
            self.corr_matrix = list(model_params['corr_matrix'])
        except:
            raise NameError('Parameters input are wrong')
        
        assert len(self.spot_price) == len(self.beta) == len(self.sigma), 'The number of each parameter should all be qual to the number of assets'
        self.n_assets = len(self.spot_price)

        assert len(self.corr_matrix) == self.n_assets*(self.n_assets-1) / 2, 'The number of the correlations must correspond to the number of assets'
        tri_corr_matrix = np.array([[0]*(i+1) + self.corr_matrix[((2*self.n_assets-(i+1))*i)//2:((2*self.n_assets-(i+1))*i)//2+self.n_assets-i-1] for i in range(self.n_assets)])
        self.corr_matrix = tri_corr_matrix + tri_corr_matrix.T + np.eye(self.n_assets)

        super().__init__(n_trials, n_intervals, self.tau)

        
    def generate_paths(self):
        '''
        generate K * N * M array, K is the number of assets, N is the number of paths, M is the number of sampling intervals 
        '''

        price_matrix = np.repeat(self.spot_price.reshape(self.n_assets, 1), self.n_trials, axis=1)
        self.price_paths = np.expand_dims(price_matrix, axis=2)

        corr_matrix = la.cholesky(self.corr_matrix)

        for _ in range(self.n_intervals):
            norm_delta = (np.random.randn(self.n_trials, self.n_assets) @ corr_matrix).reshape(self.n_trials, self.n_assets).T
            price_matrix = price_matrix + self.rate*price_matrix*self.delta_time + (price_matrix**self.beta.reshape(self.n_assets, 1)) * self.sigma.reshape(self.n_assets, 1)*norm_delta*np.sqrt(self.delta_time)
            self.price_paths = np.concatenate([self.price_paths, np.expand_dims(price_matrix, axis=2)], axis=2)

        self.price_paths = np.delete(self.price_paths, 0, axis=2)

        return self.price_paths





class MultiGeoBrownianMotion(MultiCEV):
    '''
    generate multiple simulated sample paths based on geometric brownian motion model
    '''

    def __init__(self, n_trials, n_intervals, **model_params):
        '''
        intiaialize the model parameters
        '''

        try:
            model_params['beta'] = [1] * len(model_params['spot_price'])
        except:
            raise NameError('Parameters input are wrong')
        
        super().__init__(n_trials, n_intervals, **model_params)





