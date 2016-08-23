import numpy as np
from scipy.optimize import leastsq, curve_fit


def multifit(func, datax, datay, datayerrors, p0, dataxerrors=None, iterations=1000, func_residuals=False, extra_args=None, _random_generator=np.random):
    """
    Does Monte Carlo fitting by varying the datay by datayerrors in order to estimate the error on the fitting parameters.

    :param func: The function for fitting. func takes independent variable as first parameter, dependent variable 2nd, then fitting variables next.
    :param func_residuals: (Optional) True if func calculates residuals. False if func returns values, not residuals. Default False.
    :param extra_args: (Optional) list of extra arguments to pass to func,.

    :returns: [fitted parameters, standard deviation means for the fitted parameters of all the iterations]
    """
    if func_residuals:
        errfunc = func
    else:
        # If func does not compute residuals, we need to do it ourselves.
        errfunc = lambda p, x, y, *extra: func(x,*p+extra) - y if len(extra) > 0 else func(x,*p)-y
    # Fit the data with curvefit
    args = (datax, datay)
    if extra_args is not None:
        args += extra_args
    pfit, perr = \
        leastsq(errfunc, p0, args=args,\
                        full_output=0, maxfev=10000)

    # Get the residuals
    residuals = errfunc(pfit, *args)

    s_res = np.std(residuals, ddof=1)
    ps = []
    # 100 random data sets are generated and fitted
    for i in range(iterations):
        if len(ps) < 1:
            guess = pfit
        else:
            guess = np.mean(ps, axis=0)
        if datayerrors is None:
            randomDelta = _random_generator.normal(0., s_res, len(datay))
            randomdataY = datay + randomDelta
        else:
            randomDelta =  np.array( [ \
                                _random_generator.normal(0., derr,1)[0] \
                                for derr in datayerrors ] )
            randomdataY = datay + randomDelta
        if dataxerrors is None:
            randomdataX = datax
        else:
            randomDeltaX =  np.array( [ \
                                _random_generator.normal(0., derr,1)[0] \
                                for derr in dataxerrors ] )
            randomdataX = datax + randomDeltaX
        args = (randomdataX, randomdataY)
        if extra_args is not None:
            args += extra_args
        randomfit, randomcov = \
            leastsq( errfunc, p0, args=args,\
                            full_output=0, maxfev=10000)
        ps.append( randomfit )

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)
    Nsigma = 1. # 1sigma gets approximately the same as methods above
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0,ddof=1)

    pfit = mean_pfit
    perr = err_pfit

    return pfit, perr
