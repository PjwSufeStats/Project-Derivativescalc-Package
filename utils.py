'''
File Name : utils.py
Aouthor : Junwen Peng
E-Mail : junwen.peng@163.sufe.edu.cn / jwpeng22@bu.edu

Introduction : This files contains utils functions which can facilitate the 
computation and researches, such as timer wrapper.
'''

import time

def timer(func):
    '''
    timing wrapper function, used for counting computation time
    '''

    def wrapper(*args, **kwargs):

        print('---- Computation Starts ----')
        tick = time.time()

        res = func(*args, **kwargs)

        tock = time.time()
        print('---- Computation Ends ----')
        print('Computation Time Used : {}s'.format(tock-tick))

        return res

    return wrapper
