# -*- coding: utf-8 -*-


from weakref import WeakKeyDictionary
from errors import *
import numpy as np
import pandas as pd


class Profile(object):
    """
    Profile object.
    """

    def __init__(self,
                 reference,
                 query):
        try:
            iter(reference)
        except TypeError:
            raise ProfileConstructorError("Reference must be an iterable.")
        try:
            iter(query)
        except TypeError:
            raise ProfileConstructorError("Query must be an iterable.")
        if len(reference) < len(query):
            raise ProfileLengthError("Reference longer than query")
        self.reference = tuple(reference)
        self.query = tuple(query)
        self._construct()

    def _construct(self):
        """
        Construct profile from Profile.reference and Profile.query.
        """
        self.profile = [True if i in self.reference else False for i in self.query]

    def _convert(self,
                 positive_sign,
                 negative_sign):
        """
        Convert profile to given sign.
        """
        return [positive_sign if True else negative_sign for i in self.profile]

    def to_string(self,
                  positive_sign="+",
                  negative_sign="-"):
        """
        Return profile as str.
        """
        if positive_sign is not None and negative_sign is not None:
            return "".join(self._convert(positive_sign,
                                         negative_sign))
        else:
            return str(self.profile)

    def to_list(self,
                positive_sign="+",
                negative_sign="-"):
        """
        Retrun profile as list.
        """
        if positive_sign is not None and negative_sign is not None:
            return list(self._convert(positive_sign,
                                      negative_sign))
        else:
            list(self.profile)

    def to_tuple(self,
                 positive_sign="+",
                 negative_sign="-"):
        """
        Return profile as tuple.
        """
        if positive_sign is not None and negative_sign is not None:
            return tuple(self._convert(positive_sign,
                                       negative_sign))
        else:
            return tuple(self.profile)

    def to_array(self,
                 positive_sign="+",
                 negative_sign="-"):
        """
        Return profile as an numpy.array.
        """
        if positive_sign is not None and negative_sign is not None:
            return np.array(self._convert(positive_sign,
                                          negative_sign))
        else:
            return np.array(self.profile)

    def to_series(self,
                  positive_sign="+",
                  negative_sign="-"):
        """
        Return profile as pandas.Series
        """
        if positive_sign is not None and negative_sign is not None:
            return pd.Series(self._convert(positive_sign,
                                           negative_sign))
        else:
            return pd.Series(self.profile)
