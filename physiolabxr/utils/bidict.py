from collections.abc import MutableMapping

class Bidict(MutableMapping):
    """
    A lightweight bidirectional dictionary.

    >>> b = Bidict()
    >>> b['a'] = 1
    >>> b['b'] = 2
    >>> b['a']            # forward lookup
    1
    >>> b.inv[1]          # reverse lookup
    'a'
    >>> del b['a']        # deletion keeps both views in sync
    >>> 1 in b.inv        # value disappeared on the inverse side
    False
    """
    # ------------- core API -------------------------------------------
    def __init__(self, *args, **kwargs):
        self._fwd: dict = {}
        self._rev: dict = {}
        self.update(dict(*args, **kwargs))

    # ----- MutableMapping requirements -----
    def __getitem__(self, key):
        return self._fwd[key]

    def __setitem__(self, key, val):
        # remove any existing clashes (keeps mapping 1-to-1)
        if key in self._fwd:
            old_val = self._fwd[key]
            del self._rev[old_val]
        if val in self._rev:
            old_key = self._rev[val]
            del self._fwd[old_key]

        self._fwd[key] = val
        self._rev[val] = key

    def __delitem__(self, key):
        val = self._fwd.pop(key)
        del self._rev[val]

    def __iter__(self):
        return iter(self._fwd)

    def __len__(self):
        return len(self._fwd)

    # ------------- nice extras ----------------------------------------
    def __repr__(self):
        return f"{self.__class__.__name__}({self._fwd})"

    @property
    def inv(self) -> "BidictView":
        """A *read-only* inverse view (value → key)."""
        return BidictView(self)

    def clear(self):
        self._fwd.clear()
        self._rev.clear()

    # copy() and pop() automatically inherited from MutableMapping


class BidictView(MutableMapping):
    """Read-only mirror of a Bidict’s *value → key* mapping."""
    __slots__ = ("_parent",)

    def __init__(self, parent: Bidict):
        self._parent = parent

    # read methods delegate to the parent's reverse dict
    def __getitem__(self, item):
        return self._parent._rev[item]

    def __iter__(self):
        return iter(self._parent._rev)

    def __len__(self):
        return len(self._parent._rev)

    def __contains__(self, item):
        return item in self._parent._rev

    # any mutating operation is blocked
    def __setitem__(self, key, value):
        raise TypeError("inverse view is read-only")

    def __delitem__(self, key):
        raise TypeError("inverse view is read-only")

    def __repr__(self):
        return f"{self.__class__.__name__}({self._parent._rev})"
