# @Author: Shenglong Zhang
# @Date:   2023-03-21 17:14:32
# @Last Modified by:   Shenglong Zhang
# @Last Modified time: 2023-03-21 17:14

'''
here use scipy to intepolate variables from model levels to isentropic levels. 
this will take longer time due to grid-by-grid interpolation 

speed compare:(1) metpy.interpolate.interpolate_1d is the fastest; (2) auto-defined code is the second; (3) scipy method is the lowest
test: (1) takes 0.7s; (2) takes 2.1s; (3) takes 36.0s
'''

from scipy import interpolate
import numpy as np
def iscent_interp(iscent, var, pt, vertical_dim): # 首次调用时，函数被编译为机器代码
    '''
    Parameters
    ----------
    iscent : array-like or list
        One-dimensional array of desired potential temperature surfaces
    var : array-like
        3D/4D array variables will be interpolated to each isentropic level.
    pt : array-like
        Array of potential temperature
    vertical_dim : int, optional
        The axis corresponding to the vertical in the temperature array, defaults to 0.
    Returns
    -------
    ndarray
        variable interpolated to isentropic coordinates.
    '''

    shape = list(var.shape)
    shape[vertical_dim] = np.array(iscent).size
    # zero = np.zeros((1, np.array(iscent).size, 1, 1))
    # isentlevs_nd = np.broadcast_to(zero, shape)
    isentlevs_nd = np.zeros((shape[0], shape[1],shape[2], shape[3]))
    for t in range(shape[0]):
        for i in range(shape[3]):
            for j in range(shape[2]):
                # 构建y=ax的关系，interpolate.interp1d(x, y, kind='linear）
                func = interpolate.interp1d(pt[t,:,j,i], var[t,:,j,i],kind='linear', bounds_error=False, fill_value=np.nan)
                # 是一个函数，用这个函数就可以找插值点的函数值
                isentlevs_nd[t,:,j,i] = func(np.array(iscent))           
    return isentlevs_nd