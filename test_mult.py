# Test Multiprocessing


from multiprocessing import Pool


def big_calc(*args, **kwargs):
    c = 0
    for i in range(10**8):
        c += i
    return c 



if __name__ == "__main__":

    p = Pool(6)
    biglist = p.map(big_calc, range(6))
    print(list(biglist))