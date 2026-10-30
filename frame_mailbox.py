"""One owned frame/timestamp pair, overwritten by a blocking capture reader.

The consumer never queues up old frames or waits indefinitely for the producer.
The caller must unblock the reader (e.g. stop DXCam) before joining on teardown.
"""
import threading
import time


class FrameMailbox:
    def __init__(self, reader):
        self.reader = reader
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._version = self._consumed = 0
        self._error = None
        self._published_at = self.last_arrival = None
        self._thread = threading.Thread(target=self._run, name='gd-frame-reader', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                pair = self.reader()
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            if pair is None or pair[0] is None:
                self._stop.wait(.001)
                continue
            with self._condition:
                if self._stop.is_set():
                    return
                self._latest = pair
                self._published_at = time.perf_counter()
                self._version += 1
                self._condition.notify_all()

    def take(self, wait_s=.002):
        """Latest NEW pair, or (None,None); the wait has a fixed upper budget."""
        with self._condition:
            self._condition.wait_for(lambda: self._version != self._consumed
                                     or self._error is not None or self._stop.is_set(),
                                     timeout=max(0., wait_s))
            if self._stop.is_set():
                return None, None
            if self._error is not None:
                raise self._error
            if self._version == self._consumed:
                return None, None
            self._consumed = self._version
            self.last_arrival = self._published_at
            return self._latest

    def request_stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def join(self, timeout=1.):
        self._thread.join(timeout)

    @property
    def alive(self):
        return self._thread.is_alive()
