import contextlib
import joblib
from tqdm import tqdm
import numpy as np
from collections.abc import Iterable
import numbers

numerical = (numbers.Number, np.float)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def check_type(input, types):
    if not isinstance(input, types):
        raise TypeError((f"{input} should be {types}, not {type(input)}."))


def check_iterable_type(input, types):
    if not all(isinstance(i, types) for i in input):
        raise TypeError((f"{input} elements should be {types}."))


def check_iterable_or_non_iterable_type(input, types):
    if isinstance(input, Iterable):
        check_iterable_type(input, types)
    else:
        check_type(input, types)


def check_iterable_positive(input):
    if not all(i > 0 for i in input):
        raise ValueError((f"{input} elements should be positive."))


def check_iterable_nonnegative(input):
    if not all(i >= 0 for i in input):
        raise ValueError((f"{input} elements should be nonnegative."))


def check_size(input, allow_float=True, dim=3):
    check_type(input, (list, tuple, np.ndarray))
    if len(input) != dim:
        raise ValueError(f"{input} must have {dim} dimensions")
    if allow_float:
        check_iterable_type(input, (int, np.integer, float))
    else:
        check_iterable_type(input, (int, np.integer))


def check_precomputed(input):
    check_type(input, str)
    prefix = input.split(":")[0]
    if prefix not in ["file", "s3", "gc"]:
        raise NotImplementedError("only file, s3, and gc prefixes supported")


def check_binary_path(input):
    check_iterable_type(input, str)  # ensure list of str
    for bcode in input:
        if not all(c in "01" for c in bcode):
            raise ValueError(f"Binary paths are made of 0s and 1s, not like {bin}")
