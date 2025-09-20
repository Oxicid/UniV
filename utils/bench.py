# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import time
import pstats
import cProfile
import functools

from dataclasses import dataclass


# example for use timeit: print('my_func:',timeit.timeit(lambda: my_func(),number=100))

@dataclass
class timer:
    repeats: int = 1
    text: str = ''  # TODO: Add temp text

    def __enter__(self):
        self._fix_args()
        if self.repeats != 1:
            raise ValueError('Context t manager not support repeats')
        self._start = time.perf_counter()

    def __exit__(self, *exc_info):
        self._print_time_info()
        self.reset()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            self._fix_args()
            if self.text:
                self.text = f'{self.text} {func.__name__}'
            else:
                self.text = func.__name__

            self._start = time.perf_counter()
            result = None
            for i in range(self.repeats):
                result = func(*args, **kwargs)
            self._print_time_info()
            self.reset()
            return result
        return wrapper_timer

    def __iter__(self):
        self._fix_args()
        self._start = time.perf_counter()
        for _ in range(self.repeats):
            yield
        self._print_time_info()
        self.reset()

    def _fix_args(self):
        if isinstance(self.repeats, str):
            old_name = self.text
            self.text = self.repeats
            self.repeats = old_name if isinstance(old_name, int) else 1

        if self.repeats < 1:
            raise ValueError(f"Expected 'repeats' to be at least 1, but got {self.repeats}")

    def _print_time_info(self):
        elapsed_time = (time.perf_counter() - self._start)

        if self.text:
            self.text += '. '
        if self.repeats > 1:
            print(f"{self.text}Total time: {elapsed_time:0.4f}. Avg: {(elapsed_time / self.repeats):0.4f}")
        else:
            print(f"{self.text}Elapsed time: {elapsed_time:0.4f} seconds")

    def reset(self):
        self.repeats = 1
        self.text = ''


def profile(sort_by: str = "time", lines: int = 15, strip_dirs: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s)
                if strip_dirs:
                    ps.strip_dirs()
                ps.sort_stats(sort_by)
                ps.print_stats(lines)
                print(s.getvalue())
        return inner
    return decorator
