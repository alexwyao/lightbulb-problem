import numpy
import matplotlib.pyplot as plt
import pandas as pd

def C_ij_unoptimized(S_i, S_j, a_i, a_j, r):
    res = 0
    for i_x, x in enumerate(S_i):
        for i_y, y in enumerate(S_j):
            res += a_i[i_x] * a_j[i_y] * (numpy.dot(x, y)**r)
            
    return res


def LightbulbAlman(S, n, rhod, C_fcn, theta, return_C=False):
    m = int(n**(2./3))
    g = int(n/m)
    S_i = numpy.array_split(S, m)
    C_res = []

    """
    Check if any pair of vectors in the same group
    is the correlated pair.
    Runtime: O(n^(4/3))
    """
    for i in range(m):
        for x, row1 in enumerate(S_i[i]):
            for y in range(x):
                if numpy.dot(row1, S_i[i][y]) >= rhod:
                    return i, x, i, y

    """
    Compute C_ij for all ij numpy.log2(n) times.
    """
    freqs = numpy.zeros(shape=(m, m))
    for it in range(int(numpy.log2(n))):
        a = numpy.random.choice([-1, 1], size=(n,))
        a_i = numpy.array_split(a, m)
        C = numpy.zeros(shape=(m, m))
        for i in range(m):
            for j in range(i):
                C[i,j] = C_fcn(S_i[i], S_i[j], a_i[i], a_i[j], 10)
                if return_C:
                    C_res.append(C[i,j])
                if abs(C[i,j]) > 2 * theta:
                    freqs[i,j] += 1

    # Intervention here for plotting purposes
    if return_C:
        return C_res

    """
    Compute the i,j pair with the maximum likelihood of
    containing the correlated vectors.
    """
    max_val = 0
    i_opt = j_opt = -1
    for i in range(m):
        for j in range(i):
            if freqs[i,j] > max_val:
                max_val = freqs[i,j]
                i_opt, j_opt = i, j

    assert i_opt >= 0

    """
    Find the correlated vectors in S_{i_opt}, S_{j_opt}.
    """
    for x, row1 in enumerate(S_i[i_opt]):
        for y, row2 in enumerate(S_i[j_opt]):
            if numpy.dot(row1, row2) > rhod:
                return i_opt, x, j_opt, y

    raise Exception("Assertion correlated vector in i,j is False")

def test_full_alg_error():
    r11 = numpy.random.choice([-1, 1], size=(8,4))
    numpy.testing.assert_raises(Exception, LightbulbAlman, r11, 8, 10, C_ij_unoptimized, 10)

test_full_alg_error()
