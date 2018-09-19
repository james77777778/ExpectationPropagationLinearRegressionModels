# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:13:20 2018

@author: JamesChiou
"""
'''
#########################################################################################################
# Module: epBVS.R
# Date  : January 2010
# Author: Jose Miguel Hernandez Lobato
# email : josemiguel.hernandez@uam.es
#
# Module that approximates the posterior of the parameters of the the Bayesian
# variable selection model for the linear regression problem. The procedure used to perform
# the approximation is Expectation Propagation.
#
#########################################################################################################
#
# EXPORTED FUNCTIONS: epBVS
#
#########################################################################################################
#
# The main function that should be used in this module is "epBVS". You have to call it with the arguments:
#
# 	X -> Design matrix for the regression problem.
#       Y -> Target vector for the regression problem.
#	beta -> Noise precision.
#	p0 -> Prior probability that a feature is relevant for solving the regression problem.
#	v -> Variance of the slab.
#
# "epBVS" returns the approximate distribution for the posterior as a list with components:
#
#	m -> Mean vector for the marginals of the posterior.
#	v -> Variance vector for the marginals of the posterior.
#	phi -> Vector with the marginal prbobabilities of activation of the latent variables.
#
#	t1Hat -> A list with the approximation for the likelihood.
#		mHat -> Mean vector of the factorized Gaussian approximation.
#		vHat -> Variance vector for the factorized Gaussian approximation.
#
#	t2Hat -> A list with the approximation for the spike and slab prior.
#		mHat -> Mean vector of the factorized Gaussian approximation.
#		vHat -> Variance vector for the factorized Gaussian approximation.
#		phiHat -> Parameter vector for the Bernoulli approximation.
#
#	t3Hat -> A list with the approximation for Bernoulli prior.
#		phiHat -> Parameter vector for the Bernoulli approximation.
#
#	evidence -> The approximation for the evidence given by EP.
#
'''

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import inspect
from math import pi

def callback_on_crack(x):
    print(inspect.currentframe().f_back.f_locals)
    print(x)

def invLogistic(x):
    return np.log(x / (1 - x))

def logistic(x):
    return ( 1 / (1 + np.exp(-x)) )

def epBVS(X, Y, beta = 1, p0 = 0.5, v = 1):
    
    # We find the optimal configuration for the hyper-parameters
    def target(params):
            
        beta, p0, v = params
        
        beta = np.exp(beta)
        p0 = logistic(p0)
        v = np.exp(v)
        
        ret = epBVSinternal(X, Y, beta, p0, v)
        return -ret['evidence']
    
	# We call the optimization method
    
	#x0 = c(log(beta), invLogistic(p0), log(v))
    x0 = [np.log(beta), invLogistic(p0), np.log(v)]
    #ret = minimize(target, x0, method='Nelder-Mead', options={'maxiter':40 ,'disp': True}, callback=callback_on_crack)
    ret = minimize(target, x0, method='Nelder-Mead', options={'maxiter':100 ,'disp': True})
    ret = epBVSinternal(X, Y, np.exp(ret.x[ 0 ]), logistic(ret.x[ 1 ]), np.exp(ret.x[ 2 ]))

    return ret

def epBVSinternal(X, Y, beta = 1, p0 = 0.5, v = 1):
    
    d = X.shape[1]
    n = X.shape[0]
    X = X.values
    Y = Y.values
	
    # Precomputation

    tXX = np.dot(X.T,X)
    tXY = np.dot(Y.T,X)

	# We initialize the approximation

    t1Hat = {'mHat' : np.zeros(d), 'vHat' : np.full(d, np.inf)}
    t2Hat = {'mHat' : np.zeros(d), 'vHat' : np.full(d, np.inf), 'phiHat' :  np.zeros(d)}
    t3Hat = {'phiHat' : np.zeros(d)}
    a = {'m' : np.zeros(d), 'v' : np.full(d, np.inf), 'phi' : np.zeros(d), 'p' : np.full(d, np.nan), 't1Hat' : t1Hat, 't2Hat' : t2Hat, 't3Hat' : t3Hat, 'indexNegative' : None}

	# We process the approximate term for the Bernoulli prior

    a['t3Hat']['phiHat'] = np.full(d, np.log(p0 / (1 - p0)))

	# We process the approximate term for the spike and slab prior

    a['t2Hat']['vHat'] = np.full(d, p0 * v)

	# Main loop of ep. Repeated till the algorithm reaches convegence

    i = 1
    damping = 0.99
    convergence = False
    while(~ convergence and i < 1000) :

        aOld = a.copy()
        
        # We refine the approximate term for the likelihood

        if (d > n):
            inverseWoodbury = np.linalg.inv( np.diag(np.full(n, beta**-1)) + np.dot( (X * np.repeat(a['t2Hat']['vHat'].reshape(-1,1), n, axis=1).T) , X.T) )
            vectorAux = a['t2Hat']['vHat']**-1 * a['t2Hat']['mHat'] + beta * tXY
            a['m'] = a['t2Hat']['vHat'].reshape(-1,1) * (vectorAux.reshape(-1,1) - np.dot(X.T, np.dot(inverseWoodbury , np.dot(X , (a['t2Hat']['vHat'] * vectorAux).reshape(-1,1) ))))
            a['v'] = a['t2Hat']['vHat'] - a['t2Hat']['vHat']**2 * np.dot( np.full(n,1), (X * np.dot(inverseWoodbury, X) ) )
        else :
            Sigma = np.linalg.inv(np.diag( (a['t2Hat']['vHat']**-1).ravel()) + beta * tXX)
            a['m'] = np.dot(Sigma , ( a['t2Hat']['vHat']**-1 * a['t2Hat']['mHat'] + beta * tXY).reshape(-1,1))
            a['v'] = np.diag(Sigma)

        a['t1Hat']['mHat'] = (damping * (( a['m'].T * a['v']**-1 ) - a['t2Hat']['mHat'] * a['t2Hat']['vHat']**-1) + (1 - damping) * (a['t1Hat']['mHat'] * a['t1Hat']['vHat']**-1)) 
        a['t1Hat']['vHat'] = 1 / (damping * (1 / a['v'] - 1 / a['t2Hat']['vHat']) + (1 - damping) / a['t1Hat']['vHat'])
        a['t1Hat']['mHat'] = a['t1Hat']['vHat'] * a['t1Hat']['mHat']

        damping = damping * 0.99

		# We refine the approximate term for the spike and slab prior
        
        phiHatNew = 0.5 * np.log(a['t1Hat']['vHat']) - 0.5 * np.log(a['t1Hat']['vHat'] + v) + 0.5 * a['t1Hat']['mHat']**2 * (a['t1Hat']['vHat']**-1 - (a['t1Hat']['vHat'] + v)**-1)
        
        aa = logistic(phiHatNew + a['t3Hat']['phiHat']) * a['t1Hat']['mHat'] * (a['t1Hat']['vHat'] + v)**-1 +\
                logistic(-phiHatNew - a['t3Hat']['phiHat']) * a['t1Hat']['mHat'] * a['t1Hat']['vHat']**-1
        bb = logistic(phiHatNew + a['t3Hat']['phiHat']) * (a['t1Hat']['mHat']**2 - a['t1Hat']['vHat'] - v) * (a['t1Hat']['vHat'] + v)**-2 +\
                logistic(-phiHatNew -  a['t3Hat']['phiHat']) * (a['t1Hat']['mHat']**2 * a['t1Hat']['vHat']**-2 - a['t1Hat']['vHat']**-1)
        vHatNew = (aa**2 - bb)**-1 - a['t1Hat']['vHat']
        mHatNew = a['t1Hat']['mHat'] - aa * (vHatNew + a['t1Hat']['vHat'])

        a['indexNegative'] = np.where(vHatNew < 0)

		# We minimize the KL divergence with vHatNew constrained to be positive.
        
        vHatNew[ a['indexNegative'] ] = 100
        temp = a['t1Hat']['vHat'].reshape(1,-1)
        mHatNew[ a['indexNegative'] ] = a['t1Hat']['mHat'][ a['indexNegative'] ] - aa[ a['indexNegative'] ] * \
			(vHatNew[ a['indexNegative'] ] + temp[ a['indexNegative'] ])

        a['t2Hat']['phiHat'] = phiHatNew * damping + a['t2Hat']['phiHat'] * (1 - damping)
        a['t2Hat']['mHat'] = damping * mHatNew * vHatNew**-1 + (1 - damping) * a['t2Hat']['mHat'] * a['t2Hat']['vHat']**-1
        
        a['t2Hat']['vHat'] = 1 / (damping / vHatNew + (1 - damping) / a['t2Hat']['vHat'])
        a['t2Hat']['mHat'] = a['t2Hat']['mHat'] * a['t2Hat']['vHat']

        # We compute the posterior approximation from the approximate terms

        a['v'] = 1 / (1 / a['t1Hat']['vHat'] + 1 / a['t2Hat']['vHat'])
        a['m'] = a['v'] * (a['t1Hat']['mHat'] / a['t1Hat']['vHat'] + a['t2Hat']['mHat'] / a['t2Hat']['vHat'])
        a['phi'] = a['t2Hat']['phiHat'] + a['t3Hat']['phiHat']
        a['p'] = logistic(a['phi'])

        convergence = checkConvergence(a, aOld)

        i = i + 1


	# We compute the evidence

    a['evidence'] = computeEvidence(a, Y, X, beta, v)

    a['beta'] = beta
    a['p0'] = p0
    a['vSlab'] = v

    a['Sigma'] = np.linalg.inv( np.diag(a['t2Hat']['vHat']**-1) + (beta * tXX) )

    # We return the current approximation

    return a

##
# Checks convergence of the EP algorithm.
#
# Input:
# 	a    -> The previous approximation.
# 	aNew -> The new approximation.
# Output:
# 	TRUE if the values in aOld are differ from those in aNew by less than a small constant.
#
def checkConvergence(a, aOld):

    tol = 1e-4

    convergence = np.max(np.max(np.abs(a['m'] - aOld['m'])))
    convergence = np.max([convergence, np.max(np.abs(a['v'] - aOld['v']))])

    #print(convergence)

    if (convergence < tol):
        return True
    else:
        return False

##
# Function that computes the log evidence
#
def computeEvidence(a, Y, X, beta, v):

    d = X.shape[1]
    n = X.shape[0]

    # We compute the logarithm of s1 and s2

    if (n > d):
        alpha = np.linalg.det(np.dot(np.dot(np.diag(a['t2Hat']['vHat'].ravel()) , X.T) , X) * beta + np.diag(np.ones(d)))
    else: # matrix(a$t2Hat$vHat, n, d, byrow = T)
        alpha = np.linalg.det(np.dot((np.repeat(a['t2Hat']['vHat'], n, axis=0) * X) , X.T) * beta + np.diag(np.ones(n)))

    logs1 = -n / 2 * np.log(2 * pi / beta) - 0.5 * beta * np.sum(Y**2) +\
        0.5 * np.sum((a['t2Hat']['vHat']**-1 * a['t2Hat']['mHat'] + np.dot(X.T , Y) * beta) * a['m']) -0.5 * np.sum(a['t2Hat']['mHat']**2 * a['t2Hat']['vHat']**-1) - 0.5 * np.log(alpha) +\
        1 / 2 * np.sum(np.log(1 + a['t2Hat']['vHat'] * a['t1Hat']['vHat']**-1)) +\
        1 / 2 * np.sum(a['t2Hat']['mHat']**2 * a['t2Hat']['vHat']**-1 + a['t1Hat']['mHat']**2 * a['t1Hat']['vHat']**-1 - a['m']**2 * a['v']**-1)

    c = logistic(a['t3Hat']['phiHat']) * norm(a['t1Hat']['mHat'],np.sqrt(a['t1Hat']['vHat'] + v)).pdf(0) + logistic(-a['t3Hat']['phiHat']) *\
        norm(a['t3Hat']['phiHat'], np.sqrt(a['t1Hat']['vHat'])).pdf(0)

    logs2 = np.sum(np.log(c) + 1 / 2 * np.log(1 + a['t1Hat']['vHat'] * a['t2Hat']['vHat']**-1) +\
                 1 / 2 * (a['t2Hat']['mHat']**2 * a['t2Hat']['vHat']**-1 + a['t1Hat']['mHat']**2 * a['t1Hat']['vHat']**-1 - a['m']**2 * a['v']**-1) +\
                 np.log(logistic(a['phi']) / logistic(a['t3Hat']['phiHat']) + logistic(-a['phi']) / logistic(-a['t3Hat']['phiHat'])))

    aux = d / 2 * np.log(2 * pi) + 0.5 * np.sum(np.log(a['v'])) - 0.5 * np.sum(a['t1Hat']['mHat']**2 / a['t1Hat']['vHat']) -\
		0.5 * np.sum(a['t2Hat']['mHat']**2 / a['t2Hat']['vHat']) + 0.5 * np.sum(a['m']**2 / a['v'])

    return ( logs1 + logs2 + aux + np.sum(np.log(logistic(a['t2Hat']['phiHat']) * logistic(a['t3Hat']['phiHat']) + logistic(-a['t2Hat']['phiHat']) * logistic(-a['t3Hat']['phiHat']))) )
