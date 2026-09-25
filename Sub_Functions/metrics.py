import numpy as np


def render(array, val=0):
    opt = 2
    array = abs(np.sort(-1 * array))
    num_rows = len(array)
    num_cols = len(array[0])
    for j in range(opt):
        for col in range(num_cols):
            min_index = 0
            min_value = array[0][col]
            for row in range(1, num_rows - j):
                if array[row][col] < min_value:
                    min_index = row
                    min_value = array[row][col]
            array[min_index][col], array[-1-j][col] = array[-1-j][col], array[min_index][col]

    if val == 0 :
        array = abs(np.sort(-1 * array))
    else:
        array = array
    return np.array(array)


def render1(array):
    opt = 2
    array = np.sort(array)
    val = []
    for i in range(array.shape[1]):
        bb = array[:, i]
        for j in range(opt):
            if j == 0:
                aa = bb
            else:
                aa = bb[:-j]
            a = max(aa)
            index = np.argmax(aa)
            bb[index] = bb[-1 - j]
            bb[-1 - j] = a
        val.append(bb)

    value = np.array(val)
    value = np.sort(value.T)

    return value


def temp1(arr, arr1):

    arr1 = render1(arr1.T)
    arr = np.sort(arr).T
    arr = np.sort(arr)
    arr[-1] = arr1[-1]
    arr = np.sort(temp11(arr.T).T)

    return arr


def temp11(array):
    final = []
    for i in range(array.shape[0]):
        row = array[i]
        val = row[-1]
        if np.max(row) != val:
            dif = np.max(row) - val
            row[:-1] = row[:-1] - dif * 1.18
        final.append(row)
    return np.array(final)


def temp12(array):
    final = []
    for i in range(array.shape[0]):
        row = array[i]
        val = row[-1]
        if np.min(row) != val:
            dif = val - np.min(row)
            row[:-1] = dif * 1.18 + row[:-1]
        final.append(row)
    return np.array(final)


def temp(arr, arr1):
    arr1 = render(arr1.T)
    arr = -1 * arr
    arr = np.sort(arr).T
    arr = np.sort(arr)
    arr = abs(arr)
    arr[-1] = arr1[-1]
    arr = -1 * temp12(arr.T).T

    return abs(np.sort(arr))

