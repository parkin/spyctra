import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import sys

def arPLS(y, lambda_=5.e5, ratio=1.e-6, itermax=50, log=False):
    """
    Baseline correction using asymmetrically reweighted penalized least squares
    smoothing.

    http://pubs.rsc.org/en/Content/ArticleLanding/2015/AN/C4AN01061B#!divAbstract

    Usage:
    >>> from spyctra import arPLS
    >>> # y is a 1D spectrum
    >>> background_subtracted = arPLS(y)

    :param y: The 1D spectrum
    :param lambda_: (Optional) Adjusts the balance between fitness and smoothness.
                    A smaller lamda_ favors fitness.
                    Default is 1.e5.
    :param ratio: (Optional) Iteration will stop when the weights stop changing.
                    (weights_(i) - weights_(i+1)) / (weights_(i)) < ratio.
                    Default is 1.e-6.
    :param log: (Optional) True to debug log. Default False.
    :returns: The smoothed baseline of y.
    """
    y = np.array(y)

    N = y.shape[0]

    E=eye(N,format='csc')
    # numpy.diff() does not work with sparse matrix. This is a workaround.
    # It creates the second order difference matrix.
    # [1 -2 1 ......]
    # [0 1 -2 1 ....]
    # [.............]
    # [.... 0 1 -2 1]
    D=E[:-2]-2*E[1:-1]+E[2:]

    H = lambda_*D.T*D
    Y = np.matrix(y)

    w = np.ones(N)

    for i in range(itermax+10):
        W=diags(w,0,shape=(N,N))
        Q = W+H
        B = W*Y.T

        z=spsolve(Q,B)
        d = y-z
        dn = d[d<0.0]

        m = np.mean(dn)
        if np.isnan(m):
            # add a tiny bit of noise to Y
            y2 = y.copy()
            if np.std(y) != 0.:
                y2 += (np.random.random(y.size)-0.5)*np.std(y)/1000.
            elif np.mean(y) != 0.0 :
                y2 += (np.random.random(y.size)-0.5)*np.mean(y)/1000.
            else:
                y2 += (np.random.random(y.size)-0.5)/1000.
            y = y2
            Y = np.matrix(y2)
            W=diags(w,0,shape=(N,N))
            Q = W+H
            B = W*Y.T

            z=spsolve(Q,B)
            d = y-z
            dn = d[d<0.0]

            m = np.mean(dn)
        s = np.std(dn,ddof=1)

        wt = 1./(1 + np.exp(2. * (d - (2*s-m))/s))

        # check exit condition
        condition = np.linalg.norm(w-wt) / np.linalg.norm(w)
        if condition < ratio:
            break
        if i > itermax:
            if log:
                sys.stderr.write("\nSURPASSED ITERMAX: {0}\tCondition: {1}\n".format(i, condition))
            break

        w = wt

    return z

def arPLS2d(R, lambda_=5.e5, ratio=1.e-6, itermax=50, log=False):
    """
    :param R: NxN matrix to be smoothed.
    """
    N = R.shape[0]

    # flatten the array
    Y = np.matrix(R.flatten())
    N2 = N**2

    E = eye(N2, format='csc')

    # There will be N-2 blocks in the D matrix
    # Each block will be N-2 rows
    blocks = N-2
    ### Create the difference matrix,
    ### will be (N-2)^2 x N
    D = csc_matrix((blocks**2, N))
    for i in xrange(blocks):
        offset = 2 * (i+1) + blocks*i
        D[i*blocks:(i+1)*blocks] += E[offset:offset+blocks]
        offset2 = offset + 2 + blocks
        D[i*blocks:(i+1)*blocks] -= 2*E[offset2:offset2+blocks]
        offset3 = offset2 + blocks
        D[i*blocks:(i+1)*blocks] += E[offset3:offset3+blocks] - 2*E[offset3+1:offset3+blocks+1] + 2*E[offset3+2:offset3+blocks+2]

    w = np.ones(N2)

    for i in range(itermax+10):
        W=diags(w,0,shape=(N2,N2))
        Q = W+H
        B = W*Y.T

        z = spsolve(Q,B)
        d = y - z
        dn = d[d<0]

        m = np.mean(dn)
        s = np.std(dn,ddof=1)

        wt = 1./(1 + np.exp(2. * (d - (2*s-m))/s))

        # check exit condition
        condition = np.linalg.norm(w-wt) / np.linalg.norm(w)

        if condition < ratio:
            break
        if i > itermax:
            if log:
                sys.stderr.write("\nSURPASSED ITERMAX: {0}\tCondition: {1}\n".format(i, condition))
            break

        w = wt

    return z.reshape((N,N))
