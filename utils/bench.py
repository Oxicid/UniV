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
                    func(*args, **kwargs)
        return wrapper_timer

def profile(func):
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
