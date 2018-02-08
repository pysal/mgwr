#Bandwidth optimization methods

__author__ = "Taylor Oshan"

import numpy as np
from copy import deepcopy
import copy
from collections import namedtuple

def golden_section(a, c, delta, function, tol, max_iter, int_score=False):
    """
    Golden section search routine
    Method: p212, 9.6.4
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    a               : float
                      initial max search section value
    b               : float
                      initial min search section value
    delta           : float
                      constant used to determine width of search sections
    function        : function
                      obejective function to be evaluated at different section
                      values
    int_score       : boolean
                      False for float score, True for integer score
    tol             : float
                      tolerance used to determine convergence
    max_iter        : integer
                      maximum iterations if no convergence to tolerance

    Returns
    -------
    opt_val         : float
                      optimal value
    opt_score       : kernel
                      optimal score
    output          : list of tuples
                      searching history
    """
    b = a + delta * np.abs(c-a)
    d = c - delta * np.abs(c-a)
    score = 0.0
    diff = 1.0e9
    iters  = 0
    output = []
    while np.abs(diff) > tol and iters < max_iter:
        iters += 1
        if int_score:
          b = np.round(b)
          d = np.round(d)

        score_a = function(a)
        score_b = function(b)
        score_c = function(c)
        score_d = function(d)

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c-a)
            #if int_score:
                #b = np.round(b)
        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c-a)
            #if int_score:
                #d = np.round(b)

        #if int_score:
        # opt_val = np.round(opt_val)
        output.append((opt_val, opt_score))
        diff = score_b - score_d
        score = opt_score
    return np.round(opt_val, 2), opt_score, output

def equal_interval(l_bound, u_bound, interval, function, int_score=False):
    """
    Interval search, using interval as stepsize

    Parameters
    ----------
    l_bound         : float
                      initial min search section value
    u_bound         : float
                      initial max search section value
    interval        : float
                      constant used to determine width of search sections
    function        : function
                      obejective function to be evaluated at different section
                      values
    int_score       : boolean
                      False for float score, True for integer score

    Returns
    -------
    opt_val         : float
                      optimal value
    opt_score       : kernel
                      optimal score
    output          : list of tuples
                      searching history
    """
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a,0)
        c = np.round(c,0)
        b = np.round(b,0)

    output = []

    score_a = function(a)
    score_c = function(c)

    output.append((a,score_a))
    output.append((c,score_c))

    if score_a < score_c:
        opt_val = a
        opt_score = score_a
    else:
        opt_val = c
        opt_score = score_c

    while b < c:
        score_b = function(b)

        output.append((b,score_b))

        if score_b < opt_score:
            opt_val = b
            opt_score = score_b
        b = b + interval

    return opt_val, opt_score, output

MGWR_BW_Result = namedtuple('MGWR_BW_RESULT', ['bws_','bw_trace', 'kernel_values', 'scores',
                                               'partial_predictions','model_residuals_',
                                               'partial_residuals_', 'objective_functions'])

def multi_bw(init, y, X, n, k, family, tol, max_iter, rss_score,
        gwr_func, bw_func, sel_func):
    if init:
        bw = sel_func(bw_func(y, X))
        optim_model = gwr_func(y, X, bw)
        err = optim_model.resid_response.reshape((-1,1))
        est = optim_model.params
    else:
        model = GLM(y, X, family=self.family, constant=False).fit()
        err = model.resid_response.reshape((-1,1))
        est = np.repeat(model.params.T, n, axis=0)


    XB = np.multiply(est, X)
    if rss_score:
        rss = np.sum((err)**2)
    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    VALs = []
    FUNCs = []
    try:
        from tqdm import tqdm #if they have it, let users have a progress bar
    except ImportError:
        def tqdm(x): #otherwise, just passthrough the range
            return x
    for iters in tqdm(range(1, max_iter+1)):
        new_XB = np.zeros_like(X)
        bws = []
        vals = []
        funcs = []
        current_partial_residuals = []
        ests = np.zeros_like(X)
        f_XB = XB.copy()
        f_err = err.copy()
        for i in range(k):
            temp_y = XB[:,i].reshape((-1,1))
            temp_y = temp_y + err
            temp_X = X[:,i].reshape((-1,1))
            bw_class = bw_func(temp_y, temp_X)
            funcs.append(bw_class._functions)
            bw = sel_func(bw_class)
            optim_model = gwr_func(temp_y, temp_X, bw)
            err = optim_model.resid_response.reshape((-1,1))
            est = optim_model.params.reshape((-1,))

            new_XB[:,i] = np.multiply(est, temp_X.reshape((-1,)))
            bws.append(copy.deepcopy(bw))
            ests[:,i] = est
            vals.append(bw_class.bw[1])
            current_partial_residuals.append(err.copy())

        predy = np.sum(np.multiply(ests, X), axis=1).reshape((-1,1))
        num = np.sum((new_XB - XB)**2)/n
        den = np.sum(np.sum(new_XB, axis=1)**2)
        score = (num/den)**0.5
        XB = new_XB

        if rss_score:
            new_rss = np.sum((y - predy)**2)
            score = np.abs((new_rss - rss)/new_rss)
            rss = new_rss
        scores.append(copy.deepcopy(score))
        delta = score
        BWs.append(copy.deepcopy(bws))
        VALs.append(copy.deepcopy(vals))
        FUNCs.append(copy.deepcopy(funcs))
        if delta < tol:
            break

    opt_bws = BWs[-1]
    return MGWR_BW_Result(opt_bws, np.array(BWs), np.array(VALs), 
                          np.array(scores), f_XB, f_err, current_partial_residuals, FUNCs)
