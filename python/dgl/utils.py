"""Utility module."""
from __future__ import absolute_import

from collections import Mapping
from functools import wraps
import numpy as np

import dgl.backend as F
from dgl.backend import Tensor, SparseTensor

def is_id_tensor(u):
    """Return whether the input is a supported id tensor."""
    return isinstance(u, Tensor) and F.isinteger(u) and len(F.shape(u)) == 1

def is_id_container(u):
    """Return whether the input is a supported id container."""
    return (getattr(u, '__iter__', None) is not None
            and getattr(u, '__len__', None) is not None)

class Index(object):
    """Index class that can be easily converted to list/tensor."""
    def __init__(self, data):
        self._list_data = None
        self._tensor_data = None
        self._ctx_data = dict()
        self._dispatch(data)

    def _dispatch(self, data):
        if is_id_tensor(data):
            self._tensor_data = data
        elif is_id_container(data):
            self._list_data = data
        else:
            try:
                self._list_data = [int(data)]
            except:
                raise TypeError('Error index data: %s' % str(x))

    def tolist(self):
        if self._list_data is None:
            self._list_data = list(F.asnumpy(self._tensor_data))
        return self._list_data

    def totensor(self, ctx=None):
        if self._tensor_data is None:
            self._tensor_data = F.tensor(self._list_data, dtype=F.int64)
        if ctx is None:
            return self._tensor_data
        if ctx not in self._ctx_data:
            self._ctx_data[ctx] = F.to_context(self._tensor_data, ctx)
        return self._ctx_data[ctx]

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        if self._list_data is not None:
            return len(self._list_data)
        else:
            return len(self._tensor_data)

    def __getitem__(self, i):
        return self.tolist()[i]

def toindex(x):
    return x if isinstance(x, Index) else Index(x)

def node_iter(n):
    """Return an iterator that loops over the given nodes.

    Parameters
    ----------
    n : iterable
        The node ids.
    """
    return iter(n)

def edge_iter(u, v):
    """Return an iterator that loops over the given edges.

    Parameters
    ----------
    u : iterable
        The src ids.
    v : iterable
        The dst ids.
    """
    if len(u) == len(v):
        # many-many
        for uu, vv in zip(u, v):
            yield uu, vv
    elif len(v) == 1:
        # many-one
        for uu in u:
            yield uu, v[0]
    elif len(u) == 1:
        # one-many
        for vv in v:
            yield u[0], vv
    else:
        raise ValueError('Error edges:', u, v)

def edge_broadcasting(u, v):
    """Convert one-many and many-one edges to many-many.

    Parameters
    ----------
    u : Index
        The src id(s)
    v : Index
        The dst id(s)

    Returns
    -------
    uu : Index
        The src id(s) after broadcasting
    vv : Index
        The dst id(s) after broadcasting
    """
    if len(u) != len(v) and len(u) == 1:
        u = toindex(F.broadcast_to(u.totensor(), v.totensor()))
    elif len(u) != len(v) and len(v) == 1:
        v = toindex(F.broadcast_to(v.totensor(), u.totensor()))
    else:
        assert len(u) == len(v)
    return u, v

'''
def convert_to_id_container(x):
    if is_id_container(x):
        return x
    elif is_id_tensor(x):
        return F.asnumpy(x)
    else:
        try:
            return [int(x)]
        except:
            raise TypeError('Error node: %s' % str(x))
    return None

def convert_to_id_tensor(x, ctx=None):
    if is_id_container(x):
        ret = F.tensor(x, dtype=F.int64)
    elif is_id_tensor(x):
        ret = x
    else:
        try:
            ret = F.tensor([int(x)], dtype=F.int64)
        except:
            raise TypeError('Error node: %s' % str(x))
    ret = F.to_context(ret, ctx)
    return ret
'''

class LazyDict(Mapping):
    """A readonly dictionary that does not materialize the storage."""
    def __init__(self, fn, keys):
        self._fn = fn
        self._keys = keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise KeyError(key)
        return self._fn(key)

    def __contains__(self, key):
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

class ReadOnlyDict(Mapping):
    """A readonly dictionary wrapper."""
    def __init__(self, dict_like):
        self._dict_like = dict_like

    def keys(self):
        return self._dict_like.keys()

    def __getitem__(self, key):
        return self._dict_like[key]

    def __contains__(self, key):
        return key in self._dict_like

    def __iter__(self):
        return iter(self._dict_like)

    def __len__(self):
        return len(self._dict_like)

def build_relabel_map(x):
    """Relabel the input ids to continuous ids that starts from zero.

    Parameters
    ----------
    x : Index
      The input ids.

    Returns
    -------
    new_to_old : tensor
      The mapping from new id to old id.
    old_to_new : tensor
      The mapping from old id to new id. It is a vector of length MAX(x).
      One can use advanced indexing to convert an old id tensor to a
      new id tensor: new_id = old_to_new[old_id]
    """
    x = x.totensor()
    unique_x, _ = F.sort(F.unique(x))
    map_len = int(F.max(unique_x)) + 1
    old_to_new = F.zeros(map_len, dtype=F.int64)
    # TODO(minjie): should not directly use []
    old_to_new[unique_x] = F.astype(F.arange(len(unique_x)), F.int64)
    return unique_x, old_to_new

def build_relabel_dict(x):
    """Relabel the input ids to continuous ids that starts from zero.

    The new id follows the order of the given node id list.

    Parameters
    ----------
    x : list
      The input ids.

    Returns
    -------
    relabel_dict : dict
      Dict from old id to new id.
    """
    relabel_dict = {}
    for i, v in enumerate(x):
        relabel_dict[v] = i
    return relabel_dict

class CtxCachedObject(object):
    """A wrapper to cache object generated by different context.

    Note: such wrapper may incur significant overhead if the wrapped object is very light.

    Parameters
    ----------
    generator : callable
        A callable function that can create the object given ctx as the only argument.
    """
    def __init__(self, generator):
        self._generator = generator
        self._ctx_dict = {}

    def get(self, ctx):
        if not ctx in self._ctx_dict:
            self._ctx_dict[ctx] = self._generator(ctx)
        return self._ctx_dict[ctx]

def ctx_cached_member(func):
    """Convenient class member function wrapper to cache the function result.

    The wrapped function must only have two arguments: `self` and `ctx`. The former is the
    class object and the later is the context. It will check whether the class object is
    freezed (by checking the `_freeze` member). If yes, it caches the function result in
    the field prefixed by '_CACHED_' before the function name.
    """
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self, ctx):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                bind_func = lambda _ctx : func(self, _ctx)
                setattr(self, cache_name, CtxCachedObject(bind_func))
            return getattr(self, cache_name).get(ctx)
        else:
            return func(self, ctx)
    return wrapper

def cached_member(func):
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                setattr(self, cache_name, func(self))
            return getattr(self, cache_name)
        else:
            return func(self)
    return wrapper
