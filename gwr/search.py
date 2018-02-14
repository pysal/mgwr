"""
Bandwidth optimization methods
"""

__author__ = "Taylor Oshan"

import numpy as np

def golden_section(a, c, delta, function, tol, max_iter, int_score=True):
    b = a + delta * np.abs(c-a)
    d = c - delta * np.abs(c-a)
    score = 0.0
    diff = 1.0e9
    iters  = 0
    #output = []
    dict = {}
    for iters in range(max_iter):
        if np.abs(diff) <= tol:
            break
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)
        #score_a = function(a)
        
        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
        
        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d
        
        '''
            pool = Pool(4)
            data_list = [b,d,b,d]
            result_list = pool.map(fitGWR_Ziqi, data_list)
            
            #print(data_list)
            score_b = result_list[0]
            score_d = result_list[1]
            '''
        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c-a)
        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c-a)
        #output.append((opt_val, opt_score))
        diff = score_b - score_d
        score = opt_score
    print(iters,np.round(opt_val, 2), opt_score)
    return np.round(opt_val, 2), opt_score


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

