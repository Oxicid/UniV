"""
Created by Oxicid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import io
import time
import pstats
import cProfile
import functools

from dataclasses import dataclass


@dataclass
class timer:
    text: str = ''
    repeat: int = 1
    name = ''

    def __enter__(self):
        self._start_time = time.perf_counter()

    def __exit__(self, *exc_info):
        elapsed_time = (time.perf_counter() - self._start_time)

        prefix = ''
        if self.name:
            prefix = self.name + '. '
        if self.text:
            prefix += self.text + '. '

        if self.repeat > 1 and self.name:
            print(f"{prefix}Total time: {elapsed_time:0.4f}. Avg time: {(elapsed_time / self.repeat):0.4f} seconds")
        else:
            print(f"{prefix}Elapsed time: {elapsed_time:0.4f} seconds")

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            self.name = func.__name__
            with self:
                for i in range(self.repeat):
                    result = func(*args, **kwargs)
                return result
        return wrapper_timer

def profile(func):
    # Source: https://osf.io/upav8
    @functools.wraps(func)
    def inner(*args, **qwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **qwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner
