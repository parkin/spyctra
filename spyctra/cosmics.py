import numpy as np
from scipy.interpolate import interp1d

def remove_cosmics(spectrum, max_curvature=-1000):
    """
    :param max_curvature: Maximum curvature to be allowed. This should be a
        negative number, indicating negative curvature.
    """
    n = len(spectrum)

    # Calculate the 2nd derivative
    curvature = np.gradient(np.gradient(spectrum))

    spec_indices = np.indices(spectrum.shape)[0]

    # loop through the high curvature points and replace with the interpolated value
    for index in spec_indices[np.where(curvature <= max_curvature)]:
        # Ignore spikes close to the ends of the spectrum, as interp1d will
        # throw an error if they are included.
        if index > 5 and index < n - 6:
            minindex = max(0, index - 10)
            maxindex = min(n, index + 10)
            # interpolate the local data, using only points that have low curvature
            where_smooth = np.where(curvature[minindex:maxindex] > max_curvature)
            temp_indices = spec_indices[minindex:maxindex][where_smooth]
            temp_spec = spectrum[minindex:maxindex][where_smooth]
            f = interp1d(temp_indices, temp_spec, kind='quadratic')

            spectrum[index] = f(index)

    return spectrum
