# -*- coding: utf-8 -*-

from functools import partial
import traceback
from requests import Session
import gevent
from gevent import monkey
from gevent.pool import Pool
monkey.patch_all(thread=False, select=False)


class AsyncRequest(object):
    def __init__(self, method, url, **kwargs):
        self.method = method
        self.url = url
        self.session = Session() if kwargs.pop('session', None) else None
        callback = kwargs.pop('callback', None)
        if callback:
            kwargs['hooks'] = {
                'response': callback
            }
        self.kwargs = kwargs
        self.response = None
        self.exception = None
        self.traceback = None

    def send(self, **kwargs):
        merged_kwargs = {}
        merged_kwargs.update(self.kwargs)
        merged_kwargs.update(kwargs)
        try:
            self.response = self.session.request(self.method, self.url, **merged_kwargs)
        except Exception as e:
            self.exception = e
            self.traceback = traceback.format_exc()
        return self


def send(r, pool=None, stream=False):
    if pool is not None:
        return pool.spawn(r.send, stream=stream)
    return gevent.spawn(r.send, stream=stream)


get = partial(AsyncRequest, 'GET')
options = partial(AsyncRequest, 'OPTIONS')
head = partial(AsyncRequest, 'HEAD')
post = partial(AsyncRequest, 'POST')
put = partial(AsyncRequest, 'PUT')
patch = partial(AsyncRequest, 'PATCH')
delete = partial(AsyncRequest, 'DELETE')


def async_request(method, url, **kwargs):
    return AsyncRequest(method, url, **kwargs)


def map_request(requests, stream=False, size=None, exception_handler=None, gtimeout=None):
    requests = list(requests)
    pool = Pool(size) if size else None
    jobs = [send(r, pool, stream=stream) for r in requests]
    gevent.joinall(jobs, timeout=gtimeout)
    ret = []
    for r in requests:
        if r.response is not None:
            ret.append(r.response)
        elif exception_handler and hasattr(r, 'exception'):
            ret.append(exception_handler(r, r.exception))
        else:
            ret.append(None)
    return ret


def imap_request(requests, stream=False, size=2, exception_handler=None):
    pool = Pool(size)

    def send(r):
        return r.send(stream=stream)

    for r in pool.imap_unordered(send, requests):
        if r.response is not None:
            yield r.response
        elif exception_handler:
            exception_handler(r, r.exception)

    pool.join()
