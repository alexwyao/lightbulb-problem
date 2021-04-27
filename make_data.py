def generate_dataset(n, d, rho, separate=False):
    assert 0 < rho <= 1
    
    r = numpy.random.choice([-1, 1], size=(n,d))
    rhod = int(numpy.ceil(rho * d))

    """
    Correlated vecs
    """
    ind2 = -1 if separate else 1
    for i in range(rhod):
        r[0, i] = 1
        r[ind2, i] = 1
        
    for i in range(rhod, d, 2):
        r[0, i] = 1
        r[ind2, i] = -1
    
    for i in range(rhod+1, d, 2):
        r[0, i] = -1
        r[ind2, i] = -1
    
    return r
