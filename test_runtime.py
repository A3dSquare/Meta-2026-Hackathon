"""Freshness, ownership, bounded waits and unchanged pixel checks."""
import queue
import threading
import time
import unittest

from frame_mailbox import FrameMailbox


class MailboxTests(unittest.TestCase):
    def setUp(self):
        self.source=queue.Queue()
        self.mailbox=FrameMailbox(self.source.get).start()

    def tearDown(self):
        self.mailbox.request_stop();self.source.put((None,None));self.mailbox.join(.5)
        self.assertFalse(self.mailbox.alive)

    def publish(self,*pairs):
        target=self.mailbox._version+len(pairs)
        for pair in pairs:self.source.put(pair)
        with self.mailbox._condition:
            self.assertTrue(self.mailbox._condition.wait_for(lambda:self.mailbox._version>=target,.5))

    def test_newest_pair_replaces_backlog_and_is_consumed_once(self):
        self.publish(('first',1.),('second',2.),('latest',3.))
        self.assertEqual(self.mailbox.take(0),('latest',3.))
        self.assertEqual(self.mailbox.take(0),(None,None))
        self.publish(('next',4.))
        self.assertEqual(self.mailbox.take(0),('next',4.))

    def test_blocked_producer_does_not_block_consumer(self):
        self.assertTrue(self.mailbox.alive)
        start=time.perf_counter()
        self.assertEqual(self.mailbox.take(.002),(None,None))
        self.assertLess(time.perf_counter()-start,.1)

    def test_stop_wakes_waiting_consumer_and_discards_late_result(self):
        self.mailbox.request_stop();self.source.put(('late',3.));self.mailbox.join(.5)
        self.assertEqual(self.mailbox.take(0),(None,None))
        self.assertEqual(self.mailbox._version,0)

    def test_source_exception_reaches_consumer_for_capture_recovery(self):
        def broken():raise RuntimeError('lost device')
        other=FrameMailbox(broken).start();other.join(.5)
        with self.assertRaisesRegex(RuntimeError,'lost device'):other.take(0)
        other.request_stop()


class CaptureMailboxTests(MailboxTests):
    def setUp(self):
        super().setUp()
        import numpy as np
        from capture import GDCapture,CaptureConfig
        from training_diagnostics import Timings
        self.np=np
        self.capture=GDCapture(CaptureConfig(prefetch=True))
        self.capture._running=True;self.capture._camera=object()
        self.capture._mailbox=self.mailbox
        self.capture._throttle_to_fps_cap=lambda:0
        self.capture._last_good_ts=time.perf_counter()
        self.capture.profile=Timings()

    def test_frame_timestamp_and_owned_pixels_stay_together(self):
        frame=self.np.zeros((180,320,3),self.np.uint8);frame[:,:,0]=255
        ts=time.perf_counter();self.publish((frame,ts))
        first=self.capture.read()
        self.assertEqual(self.capture.last_frame_ts,ts)
        self.assertTrue((first[-1,:,:,2]==255).all())
        self.assertEqual(self.capture.health['frames'],1)
        next_frame=self.np.zeros_like(frame)
        self.publish((next_frame,time.perf_counter()));self.capture.read()
        self.assertTrue((first[-1,:,:,2]==255).all())
        self.assertTrue(first.flags.c_contiguous)
        self.assertIn('capture_fetch',self.capture.profile.summary())

    def test_repeated_and_stale_frames_are_drops_not_fresh_motion(self):
        frame=self.np.zeros((180,320,3),self.np.uint8)
        ts=time.perf_counter();self.publish((frame,ts));self.capture.read()
        self.capture.read()
        self.publish((frame,ts-1.));self.capture.read()
        h=self.capture.health
        self.assertEqual((h['tick'],h['frames'],h['drops']),(3,1,2))
        self.assertEqual(self.capture.last_frame_ts,ts)


class PixelCostTests(unittest.TestCase):
    def test_capture_clock_fallback_and_close_are_safe(self):
        from capture_clock import CaptureClock
        from unittest.mock import patch
        clock=CaptureClock();clock.close();clock.close()
        with patch('capture_clock.time.sleep') as sleep:
            clock.sleep(0.);sleep.assert_not_called()
            clock.sleep(.001);sleep.assert_called_once_with(.001)

    def test_cropped_banner_check_exactly_matches_original(self):
        import numpy as np
        from gd import ProgressReward
        reward=ProgressReward();rng=np.random.default_rng(4)
        for shape in ((180,320,3),(4,180,320,3),(180,320)):
            img=rng.integers(0,256,shape,dtype=np.uint8)
            for white in (False,True):
                if white:img[:]=255
                full=img[-1] if img.ndim==4 else img
                mx=full.max(axis=2).astype(np.float32) if full.ndim==3 else full.astype(np.float32)
                mn=full.min(axis=2).astype(np.float32) if full.ndim==3 else mx
                h,w=mx.shape;c=reward.cfg
                y0,y1=int(h*c.banner_y0),int(h*c.banner_y1)
                x0,x1=int(w*c.banner_x0),int(w*c.banner_x1)
                a,b=mx[y0:y1,x0:x1],mn[y0:y1,x0:x1]
                frac=float(((a>=200)&((a-b)<=40)).mean())
                self.assertEqual(reward._banner_present(img),(frac>=c.banner_white_frac,frac))


if __name__=='__main__':unittest.main()
