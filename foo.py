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

if __name__ == '__main__':
    NUMWORKER = 16
    pool = ThreadPoolExecutor(max_workers=NUMWORKER)
    ppool = mp.Pool(processes=NUMWORKER)
    start = time.time()
    res = mp_foo(50000000)
    cost = time.time() - start
    print(cost)
