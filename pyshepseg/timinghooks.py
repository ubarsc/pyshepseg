"""
A class to support placement of timing points in a Python application.
"""
import time
import threading
import contextlib
import unittest

try:
    import numpy
except ImportError:
    numpy = None


__version__ = "1.0.0"


class Timers:
    """
    Manage multiple named timers. See interval() method for example
    usage. The makeSummaryDict() method can be used to generate
    summary statistics on the timings.

    Maintains a dictionary of pairs of start/finish times, before and
    after particular operations. These are grouped by operation names,
    and for each name, a list is accumulated of the pairs, for every
    time when this operation was carried out.

    The operation names are arbitrary strings chosen by the user at each
    point where a timer is embedded in the application code.

    Timing intervals can be safely nested, so some intervals can be
    contained with others.

    The object is thread-safe, so multiple threads can accumulate to
    the same names. The object is also pickle-able.

    Example Usage::

        timings = Timers()
        with timings.interval('walltime'):
            for i in range(count):
                # Some code with no specific timer

                with timings.interval('reading'):
                    # Code to do reading operations

                with timings.interval('computation'):
                    # Code to do computation

        summary = timings.makeSummaryDict()
        print(summary)

    The resulting summary dictionary would be something like::

        {'walltime': ['total': 12.345, 'min': 1.234, ......],
         'reading': ['total': 3.456, 'min': 0.234, ......],
         'computation': ['total': 9.123, 'min': 1.012, ......]
        }

    All times are in seconds.

    These 'with interval' blocks can be scattered through an application's
    code, all using the same timings object. The summary dictionary can be
    used to generate a report at the end of the application to present to a
    user, showing how the key parts of the application compare in time taken.

    """
    def __init__(self):
        self.pairs = {}
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def interval(self, intervalName):
        """
        Use as a context manager to time a particular named interval.

        Example::

            timings = Timers()
            with timings.interval('some_action'):
                # Code block required to perform the action

        After exit from the `with` statement, the timings object will have
        accumulated the start and end times around the code block. These
        will then contribute to the reporting of time intervals.

        """
        startTime = time.time()
        yield
        endTime = time.time()
        with self.lock:
            if intervalName not in self.pairs:
                self.pairs[intervalName] = []
            self.pairs[intervalName].append((startTime, endTime))

    def getDurationsForName(self, intervalName):
        """
        For the given interval name, turns that list of start/end times
        into a list of durations (end - start), in seconds.

        Returns the list of durations.
        """
        if intervalName in self.pairs:
            intervals = [(p[1] - p[0]) for p in self.pairs[intervalName]]
        else:
            intervals = None
        return intervals

    def merge(self, other):
        """
        Merge another Timers object into this one
        """
        with self.lock:
            for intervalName in other.pairs:
                if intervalName in self.pairs:
                    self.pairs[intervalName].extend(other.pairs[intervalName])
                else:
                    self.pairs[intervalName] = other.pairs[intervalName]

    def makeSummaryDict(self):
        """
        Make some summary statistics, and return them in a dictionary
        """
        if numpy is None:
            print("Timers.makeSummaryDict() requires numpy")
            return

        d = {}
        for name in self.pairs:
            intervals = numpy.array(self.getDurationsForName(name))
            tot = float(intervals.sum())
            minVal = float(intervals.min())
            maxVal = float(intervals.max())
            meanVal = float(intervals.mean())
            pcnt25 = float(numpy.percentile(intervals, 25))
            pcnt50 = float(numpy.percentile(intervals, 50))
            pcnt75 = float(numpy.percentile(intervals, 75))
            d[name] = {'total': tot, 'min': minVal, 'max': maxVal,
                'lowerq': pcnt25, 'median': pcnt50, 'upperq': pcnt75,
                'mean': meanVal, 'count': len(intervals)}
        return d

    def __getstate__(self):
        """
        Ensure pickleability by omitting the lock attribute
        """
        d = {}
        with self.lock:
            d.update(self.__dict__)
        d.pop('lock')
        return d

    def __setstate__(self, state):
        """
        For unpickling, add a new lock attribute
        """
        self.lock = threading.Lock()
        with self.lock:
            self.__dict__.update(state)


class AllTests(unittest.TestCase):
    """
    Run all tests
    """
    places = 2

    def test_single(self):
        t = Timers()
        with t.interval('test1'):
            time.sleep(2)
        summ = t.makeSummaryDict()
        self.assertAlmostEqual(summ['test1']['total'], 2, places=self.places)

    def test_multiple(self):
        t = Timers()
        with t.interval('test2'):
            time.sleep(1)

        with t.interval('test2'):
            time.sleep(2)

        with t.interval('test3'):
            time.sleep(2.5)

        summ = t.makeSummaryDict()
        self.assertAlmostEqual(summ['test2']['total'], 3, places=self.places)
        self.assertAlmostEqual(summ['test3']['total'], 2.5, places=self.places)

    def test_nested(self):
        t = Timers()
        with t.interval('test1'):
            time.sleep(1)
            with t.interval('test2'):
                time.sleep(2)

        summ = t.makeSummaryDict()
        self.assertAlmostEqual(summ['test1']['total'], 3, places=self.places)
        self.assertAlmostEqual(summ['test2']['total'], 2, places=self.places)


def mainCmd():
    unittest.main(module='timinghooks')


if __name__ == "__main__":
    mainCmd()
