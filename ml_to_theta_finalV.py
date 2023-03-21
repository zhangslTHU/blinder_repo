# @Author: Shenglong ZHANG
# @Date:   2023-03-20  21:00
# @Last Modified by:   Shenglong ZHANG
# @Last Modified time: 2023-03-21 17:38

'''
convert σ-coordinate/model levels to traditional isentropic levels (θ-coordinate)
this code mainly process temperature and pressure variables which build the T-logp relationship accroding to Ziv and Alpert (1994),
then convert them to isentropic levels.
if only convert others variables such as PV (not include temperature and pressure), using metpy.interpolate.interpolate_1d will be ok and faster

speed compare:(1) metpy.interpolate.interpolate_1d is the fastest; (2) this code is the second; (3) scipy method is the lowest
test: (1) takes 0.7s; (2) takes 2.1s; (3) takes 36.0s

assumption:
    pressure is calculated on isentropic surfaces by assuming that temperature varies linearly with the natural log of pressure. 
    Linear interpolation is then used in the vertical to find the pressure at each isentropic level
    Any additional arguments are assumed to vary linearly with temperature and will be linearly interpolated to the new isentropic levels

reference:
（1）[metpy:isentropic_interpolation](https://github.com/Unidata/MetPy/blob/a516a847749cad13692bbd28c54c67ef0dba1ff8/src/metpy/calc/thermo.py#L2462)
(2)[等熵面的插值方法](https://journals.ametsoc.org/view/journals/apme/33/6/1520-0450_1994_033_0694_itiiea_2_0_co_2.xml?tab_body=pdf)
(3)[牛顿迭代法newton-raphson iteration](https://zhuanlan.zhihu.com/p/266566509)
'''

import numpy as np
import xarray as xr
import metpy
import metpy.calc as mpcalc
from metpy.units import units
import metpy.constants as mpconsts
import scipy.optimize as so


## -------------------------------------------------------------------------------
def broadcast_indices(indices, shape, axis):
    """Calculate index values to properly broadcast index array within data array.
    The purpose of this function is work around the challenges trying to work with arrays of
    indices that need to be "broadcast" against full slices for other dimensions.
    See usage in `interpolate_1d` or `isentropic_interpolation`.
    """
    ret = []
    ndim = len(shape)
    for dim in range(ndim):
        if dim == axis:
            ret.append(indices)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)


def make_take(ndims, slice_dim):
    """Generate a take function to index in a particular dimension."""
    def take(indexer):
        return tuple(indexer if slice_dim % ndims == i else slice(None)  # noqa: S001
                     for i in range(ndims))
    return take


def find_bounding_indices(arr, values, axis, from_below=True):
    """Find the indices surrounding the values within arr along axis.
    Returns a set of above, below, good. Above and below are lists of arrays of indices.
    These lists are formulated such that they can be used directly to index into a numpy
    array and get the expected results (no extra slices or ellipsis necessary). `good` is
    a boolean array indicating the "columns" that actually had values to bound the desired
    value(s).
    Parameters
    ----------
    arr : array-like
        Array to search for values
    values: array-like
        One or more values to search for in `arr`
    axis : int
        Dimension of `arr` along which to search
    from_below : bool, optional
        Whether to search from "below" (i.e. low indices to high indices). If `False`,
        the search will instead proceed from high indices to low indices. Defaults to `True`.
    Returns
    -------
    above : list of arrays
        List of broadcasted indices to the location above the desired value
    below : list of arrays
        List of broadcasted indices to the location below the desired value
    good : array
        Boolean array indicating where the search found proper bounds for the desired value
    """
    # The shape of generated indices is the same as the input, but with the axis of interest
    # replaced by the number of values to search for.
    indices_shape = list(arr.shape)
    indices_shape[axis] = len(values)

    # Storage for the found indices and the mask for good locations
    indices = np.empty(indices_shape, dtype=int)
    good = np.empty(indices_shape, dtype=bool)

    # Used to put the output in the proper location
    take = make_take(arr.ndim, axis)

    # Loop over all of the values and for each, see where the value would be found from a
    # linear search
    for level_index, value in enumerate(values):
        # Look for changes in the value of the test for <= value in consecutive points
        # Taking abs() because we only care if there is a flip, not which direction.
        switches = np.abs(np.diff((arr <= value).astype(int), axis=axis))

        # Good points are those where it's not just 0's along the whole axis
        good_search = np.any(switches, axis=axis)

        if from_below:
            # Look for the first switch; need to add 1 to the index since argmax is giving the
            # index within the difference array, which is one smaller.
            index = switches.argmax(axis=axis) + 1
        else:
            # Generate a list of slices to reverse the axis of interest so that searching from
            # 0 to N is starting at the "top" of the axis.
            arr_slice = [slice(None)] * arr.ndim
            arr_slice[axis] = slice(None, None, -1)

            # Same as above, but we use the slice to come from the end; then adjust those
            # indices to measure from the front.
            index = arr.shape[axis] - 1 - switches[tuple(arr_slice)].argmax(axis=axis)

        # Set all indices where the results are not good to 0
        index[~good_search] = 0

        # Put the results in the proper slice
        store_slice = take(level_index)
        indices[store_slice] = index
        good[store_slice] = good_search

    # Create index values for broadcasting arrays
    above = broadcast_indices(indices, arr.shape, axis)
    below = broadcast_indices(indices - 1, arr.shape, axis)

    return above, below, good


def _less_or_close(a, value, **kwargs):
    r"""Compare values for less or close to boolean masks.
    Returns a boolean mask for values less than or equal to a target within a specified
    absolute or relative tolerance (as in :func:`numpy.isclose`).
    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value
    Returns
    -------
    array-like
        Boolean array where values are less than or nearly equal to value
    """
    return (a < value) | np.isclose(a, value, **kwargs)



## -------------------------------------------------------------------------------
def isentropic_interpolation(levels, pressure, temperature, *args, vertical_dim=0,
                             temperature_out=False, max_iters=50, eps=1e-6,
                             bottom_up_search=True, **kwargs):
    '''
     Parameters
    ----------
    levels : array-like (K)
        One-dimensional array of desired potential temperature surfaces
    pressure : array-like (hPa)
        array of pressure levels
    temperature : array-like (K)
        Array of temperature
    args : array-like, optional
        Any additional variables will be interpolated to each isentropic level.
    Returns
    -------
    list
        List with pressure at each isentropic level, followed by each additional
        argument interpolated to isentropic coordinates.
    Other Parameters
    ----------------
    vertical_dim : int, optional
        The axis corresponding to the vertical in the temperature array, defaults to 0.
    '''
    # iteration function to be used later
    # Calculates theta from linearly interpolated temperature and solves for pressure
    def _isen_iter(iter_log_p, isentlevs_nd, ka, a, b, pok):
        exner = pok * np.exp(-ka * iter_log_p)
        t = a * iter_log_p + b
        # Newton-Raphson iteration
        f = isentlevs_nd - t * exner
        fp = exner * (ka * t - a)
        return iter_log_p - (f / fp)
    
    pressure = np.array(pressure)
    temperature = np.array(temperature)

    slices = [np.newaxis] * temperature.ndim
    slices[vertical_dim] = slice(None)
    slices = tuple(slices)

    # Sort input data
    sort_pressure = np.argsort(pressure, axis=vertical_dim)  # 升序 (time, level, lat, lon)
    sort_pressure = np.swapaxes(np.swapaxes(sort_pressure, 0, vertical_dim)[::-1], 0,
                                vertical_dim)  # 降序， 1000hPa到1hPa
    # 虽然sort_pressure已经是一个4维数组了，但为了能够降序且选择4维数组，需要再broadcast_indices一下，生成一个包含4个4维数组的元组
    sorter = broadcast_indices(sort_pressure, temperature.shape, vertical_dim)
    levs = pressure[sorter]
    tmpk = temperature[sorter]

    levels = np.asarray(levels).reshape(-1)  # K， 一维数组(n,) Convert a list into an array
    isentlevels = levels[np.argsort(levels)]  # K，升序（小值在前，大值在后），与降序后的气压对应

    # Make the desired isentropic levels the same shape as temperature
    shape = list(temperature.shape)
    shape[vertical_dim] = isentlevels.size
    isentlevs_nd = np.broadcast_to(isentlevels[slices], shape)  # isentlevels[slices]从(n,)到(1,n,1,1)，再广播则变成(time, n, lat, lon)

    # exponent to Poisson's Equation, which is imported above
    ka = mpconsts.kappa.m_as('dimensionless')  # Rd/cp_d

    # calculate theta for each point
    pres_theta = np.array(mpcalc.potential_temperature(levs * units.hPa, tmpk * units.kelvin))  # K

    # Raise error if input theta level is larger than pres_theta max
    if np.max(pres_theta) < np.max(levels):
        raise ValueError('Input theta level out of data bounds')
    # Find log of pressure to implement assumption of linear temperature dependence on
    # ln(p)
    log_p = np.log(levs)  # hPa

    # Calculations for interpolation routine
    pok = mpconsts.P0 ** ka

    # index values for each point for the pressure level nearest to the desired theta level
    above, below, good = find_bounding_indices(pres_theta, levels, vertical_dim,
                                               from_below=bottom_up_search)

    # calculate constants for the interpolation
    a = (tmpk[above] - tmpk[below]) / (log_p[above] - log_p[below])
    b = tmpk[above] - a * log_p[above]

    # calculate first guess for interpolation
    isentprs = 0.5 * (log_p[above] + log_p[below])

    # Make sure we ignore any nans in the data for solving; checking a is enough since it
    # combines log_p and tmpk.
    good &= ~np.isnan(a)

    # iterative interpolation using scipy.optimize.fixed_point and _isen_iter defined above
    log_p_solved = so.fixed_point(_isen_iter, isentprs[good],
                                  args=(isentlevs_nd[good], ka, a[good], b[good], pok.m),
                                  xtol=eps, maxiter=max_iters)
    
     # get back pressure from log p
    isentprs[good] = np.exp(log_p_solved)

    # Mask out points we know are bad as well as points that are beyond the max pressure
    isentprs[~(good & _less_or_close(isentprs, np.max(pressure)))] = np.nan

    # create list for storing output data
    # ret = [units.Quantity(isentprs, 'hPa')]  # pressure
    ret = [isentprs]  # pressure

    # if temperature_out = true, calculate temperature and output as last item in list
    if temperature_out:
        # ret.append(units.Quantity((isentlevs_nd / ((mpconsts.P0.m / isentprs) ** ka)), 'K'))  # temperature
        ret.append((isentlevs_nd / ((mpconsts.P0.m / isentprs) ** ka)))  # temperature

    # do an interpolation for each additional argument
    if args:
        others = metpy.interpolate.interpolate_1d(isentlevels, pres_theta, *(np.array(arr)[sorter] for arr in args),
                                axis=vertical_dim, return_list_always=True)
        ret.extend(others)
    # 返回的第一个是气压pressure，第二个是温度，后面的是其它变量
    return ret



## -------------------------------------------------------------------------------
def isentropic_interpolation_as_dataset(
    levels,
    pressure,
    temperature,
    *args,
    vertical_dim,
    max_iters=50,
    eps=1e-6,
    bottom_up_search=True
):
    r"""Interpolate xarray data in isobaric coords to isentropic coords, returning a Dataset.
    Parameters
    ----------
    levels : `pint.Quantity`
        One-dimensional array of desired potential temperature surfaces
    pressure : `xarray.DataArray` (hPa)
        array of pressure levels
    temperature : `xarray.DataArray`
        Array of temperature
    args : `xarray.DataArray`, optional
        Any other given variables will be interpolated to each isentropic level. Must have
        names in order to have a well-formed output Dataset.
    max_iters : int, optional
        The maximum number of iterations to use in calculation, defaults to 50.
    eps : float, optional
        The desired absolute error in the calculated value, defaults to 1e-6.
    bottom_up_search : bool, optional
        Controls whether to search for levels bottom-up (starting at lower indices),
        or top-down (starting at higher indices). Defaults to True, which is bottom-up search.
    Returns
    -------
    xarray.Dataset
        Dataset with pressure, temperature, and each additional argument, all on the specified
        isentropic coordinates.
    """
    # Ensure matching coordinates by broadcasting
    all_args = xr.broadcast(temperature, *args)

    # Obtain result as list of Quantities

    ret = isentropic_interpolation(
        levels,
        pressure,
        temperature,
        *(arg.metpy.unit_array for arg in all_args[1:]),
        vertical_dim=vertical_dim,
        temperature_out=True,
        max_iters=max_iters,
        eps=eps,
        bottom_up_search=bottom_up_search
    )

    # Reconstruct coordinates and dims (add isentropic levels, remove isobaric levels)
    vertical_dim_name = all_args[0].metpy.find_axis_name('vertical')
    new_coords = {
        'isentropic_level': xr.DataArray(
            levels.m,
            dims=('isentropic_level',),
            coords={'isentropic_level': levels.m},
            name='isentropic_level',
            attrs={
                'units': str(levels.units),
                'positive': 'up'
            }
        ),
        **{
            key: value
            for key, value in all_args[0].coords.items()
            if key != vertical_dim_name
        }
    }
    new_dims = [
        dim if dim != vertical_dim_name else 'isentropic_level' for dim in all_args[0].dims
    ]

    # Build final dataset from interpolated Quantities and original DataArrays
    return xr.Dataset(
        {
            'pressure': (
                new_dims,
                ret[0],
                {'standard_name': 'air_pressure',
                 'units': 'hPa'
                 }
            ),
            'temperature': (
                new_dims,
                ret[1],
                {'standard_name': 'air_temperature',
                 'units': 'K'
                 }
            ),
            **{
                all_args[i].name: (new_dims, ret[i + 1], all_args[i].attrs)
                for i in range(1, len(all_args))
            }
        },
        coords=new_coords
    )