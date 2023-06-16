import pickle
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
import multiprocessing as mp

def foo(num):
    cnt = 0
    for i in range(num):
        cnt += i
    return cnt

def mt_foo(num):
    step = num // NUMWORKER
    tasks = [pool.submit(foo, step) for i in range(NUMWORKER)]
    cnt = 0
    for future in as_completed(tasks):
        cnt += future.result()
    return cnt

def mp_foo(num):
    step = [num // NUMWORKER for i in range(NUMWORKER)]
    ppool.map(foo, step)

def vis():
    import matplotlib.pylab as plt
    with open('./temp_to_prod.pkl', 'rb') as f:
        ttp = pickle.load(f)
    with open('./prod_to_temp.pkl', 'rb') as f:
        ptt = pickle.load(f)
    print(len(ptt))
    # plt.bar(ptt.keys(), ptt.values())
    # plt.xlabel('products')
    # plt.ylabel('templates')
    # plt.savefig('./ptt.png')
    print(len(ttp))
    plt.bar(ttp.keys(), ttp.values())
    plt.ylim(0, 20000)
    plt.xlabel('templates')
    plt.ylabel('products')
    plt.savefig('./ttp.png')
    plt.show()

if __name__ == '__main__':
    NUMWORKER = 16
    pool = ThreadPoolExecutor(max_workers=NUMWORKER)
    ppool = mp.Pool(processes=NUMWORKER)
    vis()
