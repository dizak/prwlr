# -*- coding: utf-8 -*-


from __future__ import print_function
import math


def isiniterable(query_iterable,
                 reference_iterable,
                 all_present=False):
    """
    Returns <True> or <False> depending on whether elements from one iterable
    are found in the second.

    Parameters
    -------
    query_iterable: list, tuple, set
        Iterable from which the elements presence is checked in reference_iterable.
    reference_iterable: list, tuple, set
        Iterable in which presence of elements from query_iterable is checked.
    all_present: bool
        If <True> return <True> only if all the elements from query_iterable are
        in reference_iterable.

    Returns
    -------
    bool
        <True> of <False> if all or any of the elements from one iterable are
        found in the another.
    """
    if all_present is True:
        return all([i in reference_iterable for i in query_iterable])
    else:
        return any([i in reference_iterable for i in query_iterable])


def remove_from_list(element,
                     iterable):
    """
    Return list without given element from that list. Conversely to built-in
    methods it is fruitful function.

    Parameters
    -------
    element: object
        Element to be removed from the list.
    iterable: list, tuple, set
        Iterable from which the element should be removed.

    Returns
    -------
    list
        Cleaned up from the element.

    """
    if element in list(iterable):
        iterable.remove(element)
    return iterable


def remove_char(string,
                iterable):
    """
    Return str without given elements from the iterable. More convenient than
    chaining the built-in replace methods.

    Parameters
    -------
    string: str
        String from which the characters from the iterable are removed.
    iterable: str, list, tuple, set
        Iterable with characters that are removed from the string.

    Returns
    -------
    str
        Without elements from the iterable.

    """
    for i in iterable:
        string = string.replace(i, "")
    return string


def all_possible_combinations_counter(subset_size,
                                      set_size):
    """
    Return a number (int) of all possible combinations of elements in size
    of a subset of a set.

    Parameters
    -------
    subset_size: int
        Size of the subset.
    set_size: int
        Size of the whole set.

    Returns
    -------
    int
        Number of all combinations.
    """
    f = math.factorial
    return f(set_size) / f(subset_size) / f(set_size - subset_size)
