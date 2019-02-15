# -*- coding: utf-8 -*-


from __future__ import print_function
from prwlr.errors import *
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist


class Profile(object):
    """
    Profile object.
    """
    _positive_sign = "+"
    _negative_sign = "-"

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
        self.reference = set(reference)
        self.query = sorted(tuple(set(query)))
        self._construct()

    def __len__(self):
        return len(self.profile)

    def __repr__(self):
        return self.to_string()

    def __lt__(self,
               other):
        return self.profile < other.profile

    def __le__(self,
               other):
        return self.profile <= other.profile

    def __gt__(self,
               other):
        return self.profile > other.profile

    def __ge__(self,
               other):
        return self.profile >= other.profile

    def __ne__(self,
               other):
        return self.profile != other.profile

    def __eq__(self,
               other):
        return self.profile == other.profile and self.query == other.query

    def __hash__(self):
        return hash(self.profile)

    def _construct(self):
        """
        Construct profile from Profile.reference and Profile.query.
        """
        self.profile = tuple(True if i in self.reference else False for i in self.query)

    def _bind(self):
        """
        Return zipped Profile.query and Profile.profile.
        """
        return tuple(zip(self.query, self.profile))

    def _convert(self,
                 positive_sign,
                 negative_sign):
        """
        Convert profile to given sign.
        """
        return tuple(positive_sign if i is True else negative_sign for i in self.profile)

    def isall(self,
              queries):
        """
        Returns <True> if all positions from queries are present in the
        profile.
        """
        if all([dict(tuple(zip(self.query, self.profile)))[i] for i in queries]):
            return True
        else:
            return False

    def isany(self,
              queries):
        """
        Return <True> if any of the positions from queries is present in the
        profile.
        """
        if any([dict(tuple(zip(self.query, self.profile)))[i] for i in queries]):
            return True
        else:
            return False

    def to_string(self,
                  positive_sign=_positive_sign,
                  negative_sign=_negative_sign):
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

    def calculate_pss(self,
                      profile,
                      ignore=None,
                      method="pairwise"):
        """
        Calculate Profiles Similarity Score.
        """
        if len(self) != len(profile):
            raise ProfileError("Different profiles' lengths")
        prof_1 = self
        prof_2 = profile
        if ignore:
            for i in ignore:
                try:
                    prof_1.profile = list(prof_1.profile)
                    del prof_1.profile[prof_1.query.index(i)]
                    prof_1.profile = tuple(prof_1.profile)
                except IndexError:
                    raise ProfileError("Element to ignore not in profile")
                try:
                    prof_2.profile = list(prof_2.profile)
                    del prof_2.profile[prof_2.query.index(i)]
                    prof_2.profile = tuple(prof_2.profile)
                except IndexError:
                    raise ProfileError("Element to ignore not in profile")
        if method == "pairwise":
            return sum(a == b for a, b in zip(prof_1.profile, prof_2.profile))
        elif method == "jaccard":
            return dist.jaccard(prof_1.profile, prof_2.profile)
        elif method == "yule":
            return dist.yule(prof_1.profile, prof_2.profile)
        elif method == "dice":
            return dist.dice(prof_1.profile, prof_2.profile)
        elif method == "hamming":
            return dist.hamming(prof_1.profile, prof_2.profile)
        elif method == "kulsinski":
            return dist.kulsinski(prof_1.profile, prof_2.profile)
        elif method == "rogerstanimoto":
            return dist.rogerstanimoto(prof_1.profile, prof_2.profile)
        elif method == "russellrao":
            return dist.russellrao(prof_1.profile, prof_2.profile)
        elif method == "sokalmichener":
            return dist.sokalmichener(prof_1.profile, prof_2.profile)

    def get_present(self):
        """
        Return elements present in the profile.
        """
        return tuple(k for k, v in self._bind() if v is True)

    def get_absent(self):
        """
        Return element absent in the profile.
        """
        return tuple(k for k, v in self._bind() if v is False)
