# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:41:37 2018

@author: JamesChiou
"""

import pandas as pd
import numpy as np
import epBVS

beta = 10
p0 = 0.5
v1 = 1
nSimulations = 10

for i in range(1 , nSimulations+1):

    Xtrain = pd.read_csv('data/Xtrain'+str(i)+'.txt', sep=" ", header=None)
    Ytrain = pd.read_csv('data/Ytrain'+str(i)+'.txt', sep=" ", header=None)
    Xtest = pd.read_csv('data/Xtest'+str(i)+'.txt', sep=" ", header=None)
    Ytest = pd.read_csv('data/Ytest'+str(i)+'.txt', sep=" ", header=None)
    
    #time <- system.time(ret <- epBVS(Xtrain, Ytrain))
    ret = epBVS.epBVS(Xtrain, Ytrain)

    error = np.array( np.sqrt(np.mean((Ytest - np.dot(Xtest , ret['m'].reshape(-1,1)))**2)) )
    
    print(i)
    print( "error"+str(error) )
    print( "evidence"+str(ret['evidence']) )

    
