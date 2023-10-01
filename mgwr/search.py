# Bandwidth optimization methods

__author__ = "Taylor Oshan"

import numpy as np
from copy import deepcopy


def golden_section(a: float, c: float, delta: float,
                   function, tol: float, max_iter: int,
                   bw_max: float, int_score: bool = False,
                   verbose: bool = False) -> list:
    """
    Golden section search routine

    Method: p212, 9.6.4

    :cite:`fotheringham_geographically_2002`: Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        a (float): initial max search section value
        c (float): initial min search section value
        delta (float): constant used to determine width of search sections
        function (function): objective function to be evaluated at different section values
        int_score (bool): False for float score, True for integer score
        tol (float): tolerance used to determine convergence
        max_iter (int): maximum iterations if no convergence to tolerance
        bw_max (float): maximum bandwidth

    Returns:
        opt_val (float): optimal value
        opt_score (kernel): optimal score
        output (list of tuples): searching history
    """
    if c == np.inf:
        b = a + delta * np.abs(n - a)
        d = n - delta * np.abs(n - a)
    else:
        b = a + delta * np.abs(c - a)
        d = c - delta * np.abs(c - a)

    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    output = []
    dict_tmp = {}
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if int_score:
            b = np.round(b)
            d = np.round(d)

        if b in dict_tmp:
            score_b = dict_tmp[b]
        else:
            score_b = function(b)
            dict_tmp[b] = score_b
            if verbose:
                print(f"Bandwidth: {np.round(b, 2)}, score: {score_b[0]:.2f}")

        if d in dict_tmp:
            score_d = dict_tmp[d]
        else:
            score_d = function(d)
            dict_tmp[d] = score_d
            if verbose:
                print(f"Bandwidth: {np.round(d, 2)}, score: {score_d[0]:.2f}")

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        output.append((opt_val, opt_score))

        opt_val = np.round(opt_val, 2)
        if (opt_val, opt_score) not in output:
            output.append((opt_val, opt_score))

        diff = score_b - score_d
        # score = opt_score

    if a == np.inf or bw_max == np.inf:
        score_ols = function(np.inf)
        output.append((np.inf, score_ols))

        if score_ols <= opt_score:
            opt_score = score_ols
            opt_val = np.inf

        if verbose:
            print(f"Bandwidth: {np.inf}, score: {opt_score[0]:.2f}")

    return [opt_val, opt_score, output]


def equal_interval(l_bound: float,
                   u_bound: float,
                   interval: float,
                   function,
                   int_score: bool = False,
                   verbose: bool = False) -> list:
    """
    Interval search, using interval as step size

    Args:
        l_bound (float): initial min search section value
        u_bound (float): initial max search section value
        interval (float): constant used to determine width of search sections
        function (function): objective function to be evaluated at different section values
        int_score (bool): False for float score, True for integer score
        verbose (bool): True for printing bandwidth and score at each iteration


    Returns:
        opt_val (float): optimal value
        opt_score (kernel): optimal score
        output (list of tuples): searching history

    """
    a = l_bound
    c = u_bound
    b = a + interval
    if int_score:
        a = np.round(a, 0)
        c = np.round(c, 0)
        b = np.round(b, 0)

    score_a = function(a)
    if verbose:
        print(score_a)
        print(f"Bandwidth: {a}, score: {score_a[0]:.2f}")

    output = [(a, score_a)]
    opt_val = a
    opt_score = score_a

    while b < c:
        score_b = function(b)
        if verbose:
            print(f"Bandwidth: {b}, score: {score_b[0]:.2f}")

        output.append((b, score_b))

        if score_b < opt_score:
            opt_val = b
            opt_score = score_b
        b = b + interval

    score_c = function(c)
    if verbose:
        print(f"Bandwidth: {c}, score: {score_c[0]:.2f}")

    output.append((c, score_c))

    if score_c < opt_score:
        opt_val = c
        opt_score = score_c

    return [opt_val, opt_score, output]


def multi_bw(init, y, X, n, k, family, tol, max_iter: int, rss_score, gwr_func,
             bw_func, sel_func, multi_bw_min, multi_bw_max, bws_same_times,
             verbose=False) -> tuple:
    """
    Multiscale GWR bandwidth search procedure using iterative GAM backfitting
    """
    if init is None:
        bw = sel_func(bw_func(y, X))
        optim_model = gwr_func(y, X, bw)
    else:
        bw = init
        optim_model = gwr_func(y, X, init)
    bw_gwr = bw
    err = optim_model.resid_response.reshape((-1, 1))
    param = optim_model.params

    XB = np.multiply(param, X)
    if rss_score:
        rss = np.sum((err)**2)
    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    bw_stable_counter = 0
    bws = np.empty(k)
    gwr_sel_hist = []

    try:
        # if user have it, let users have a progress bar
        from tqdm.auto import tqdm
    except ImportError:
        # otherwise, just passthrough the range
        def tqdm(x, desc=''):
            return x

    for iters in tqdm(range(1, max_iter + 1), desc='Backfitting'):
        new_XB = np.zeros_like(X)
        params = np.zeros_like(X)

        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1))
            temp_y = temp_y + err
            temp_X = X[:, j].reshape((-1, 1))
            bw_class = bw_func(temp_y, temp_X)

            # If in backfitting,
            # all bws not changing in bws_same_times (default 5) iterations
            if bw_stable_counter >= bws_same_times:
                bw = bws[j]
            else:
                bw = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j])
                gwr_sel_hist.append(deepcopy(bw_class.sel_hist))

            optim_model = gwr_func(temp_y, temp_X, bw)
            err = optim_model.resid_response.reshape((-1, 1))
            param = optim_model.params.reshape((-1, ))
            new_XB[:, j] = optim_model.predy.reshape(-1)
            params[:, j] = param
            bws[j] = bw

        # If bws remain the same as from previous iteration
        if (iters > 1) and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0

        num = np.sum((new_XB - XB)**2) / n
        den = np.sum(np.sum(new_XB, axis=1)**2)
        score = (num / den)**0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy)**2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))

        if verbose:
            print(f"Current iteration: {iters}, SOC: {np.round(score, 7)}")
            print(f"Bandwidths: {', '.join([str(bw) for bw in bws])}")

        if delta < tol:
            break

    opt_bws = BWs[-1]
    return (opt_bws, np.array(BWs), np.array(scores), params, err, gwr_sel_hist, bw_gwr)
