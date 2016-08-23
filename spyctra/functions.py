"""
Implementation of useful functions.
"""
import numpy as np

def lorentz(p, x):
    """
    Lorentzian function.

    amplitude / (1 + ((x-xo)/gamma)**2)

    :param p: List of parameters. p[0] = gamma, p[1] = xo, p[2] = amplitude.
                If a 2D parameter list is passed in, the result will be the sum of
                multiple lorentzians.
                You can also pass in a 1D array with multiple sets of parameters, if
                so the function will return the sum of the gaussians.
    :returns: Element by element lorenzian.

    To use the single value:

    >>> x = 10
    >>> gamma = 6
    >>> xo = 9
    >>> amplitude = 20
    >>> params = [gamma, xo, amplitude]
    >>> lorentz(params, x)
    19.459459459459

    To get the sum of a 2 lorentzians of an array x:

    >>> import numpy as np
    >>> x = np.array([0., 1.])
    >>> params = [[3, 12, 33], [3, 15, 24]]
    >>> lorentz(params, x)
    array([2.8642, 3.3383])

    """
    params = np.array(p)
    x = np.array(x)

    if params.ndim < 2:
        ## Reshape a flattened array into full array
        n = len(params)
        total = n / 3
        if total > 1:
            params = params.reshape((total, 3))
        else:
            params = np.array([params])

    result = 0.

    for param in params:
        result += param[2] / (1 + np.square((1.*x - param[1])/param[0]))

    return result

def gaussian(p, x):
    """
    The Gaussian function

    amplitude * exp(- (x - mean)**2 / (2 * sigma**2))

    :param p: Parameters. p[0] = sigma, p[1] = mean, p[2] = amplitude.
            You can also pass in an 2D array of parameters, if so the function
            will return the sum of the gaussians.
            You can also pass in a 1D array with multiple sets of parameters, if
            so the function will return the sum of the gaussians.
    :param x: Single value or array of x values.
    :returns: Gaussian along x.

    To use the single value:

    >>> x = 4
    >>> sigma = 5
    >>> xo = 12
    >>> amplitude = 15
    >>> params = [sigma, xo, amplitude]
    >>> gaussian(params, x)
    4.1705595067979

    To get the sum of a 2 lorentzians of an array x:

    >>> import numpy as np
    >>> x = np.array([0., 1.])
    >>> params = [[3, 12, 33], [3, 15, 24]]
    >>> result = gaussian(params, x)

    """
    params = np.array(p)
    x = np.array(x)

    if params.ndim < 2:
        ## Reshape a flattened array into full array
        n = len(params)
        total = n / 3
        if total > 1:
            params = params.reshape((total, 3))
        else:
            params = np.array([params])

    result = 0.

    for param in params:
        result += param[2] * np.exp(- 0.5 * np.square((1.*x - param[1]) / param[0]))

    return result
