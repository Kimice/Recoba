import time


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print 'calc spend: {}s.'.format(time.time() - start)
        return result
    return wrapper
