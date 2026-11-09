"""The board relay's reading of a launch's end: a launch that completes between two register reads is a
launch, not a refusal (R41-F2's board runs l4 and l6 each logged `refused seq=N status=0` for one)."""
import io
import unittest

from tnca.backends import board_relay as br


class Device:
    """The control block's registers as the relay reads them, one read at a time. `after` reads in, the
    pending launch completes: rsp_seq and launch_sequence change together, as the RTL changes them."""

    def __init__(self, pending, *, after=None, refused_status=None):
        self.pending, self.after, self.reads = pending, after, 0
        self.regs = {br.R_SERVED: pending, br.R_LAUNCH_SEQUENCE: pending - 1, br.R_RSP_SEQ: 0,
                     br.R_RSP_STATUS: 0, br.R_LAUNCH_START: 100, br.R_LAUNCH_END: 1100,
                     br.R_RSP_R0: 990, br.R_RSP_R1: 0}
        if refused_status is not None:
            # Answered at the doorbell: rsp_seq is the request's, launch_sequence the last launch's.
            self.regs.update({br.R_RSP_SEQ: pending, br.R_RSP_STATUS: refused_status})

    def read_reg(self, offset):
        self.reads += 1
        if self.after is not None and self.reads > self.after:
            self.regs.update({br.R_RSP_SEQ: self.pending, br.R_LAUNCH_SEQUENCE: self.pending})
            self.after = None
        return self.regs[offset]


def relay_and_connection(device):
    relay = br.BoardRelay(device, '/nonexistent', image=None)
    conn = br._Connection(None, 1, 'p0', io.StringIO())
    conn.pending = device.pending
    device.reads = 0
    return relay, conn


class Observe(unittest.TestCase):
    def test_a_launch_that_completes_between_the_two_reads_is_counted_not_refused(self):
        # The first read of launch_sequence sees the launch still running; the device completes it before
        # the read of rsp_seq.
        device = Device(7, after=1)
        relay, conn = relay_and_connection(device)
        self.assertFalse(relay.observe(conn))
        self.assertEqual((conn.pending, conn.anomalies, conn.log.getvalue()), (7, 0, ''))
        self.assertTrue(relay.observe(conn))
        self.assertEqual(conn.log.getvalue(), 'cmd op=1 seq=7 obs=1000 done=990 status=0\n')
        self.assertEqual((conn.launches, conn.anomalies, conn.pending), (1, 0, None))

    def test_a_refusal_at_the_doorbell_is_still_one(self):
        device = Device(7, refused_status=3)
        relay, conn = relay_and_connection(device)
        self.assertTrue(relay.observe(conn))
        self.assertEqual(conn.log.getvalue(), 'refused seq=7 status=3\n')
        self.assertEqual((conn.launches, conn.anomalies), (0, 1))

    def test_a_running_launch_stays_pending(self):
        device = Device(7)
        relay, conn = relay_and_connection(device)
        self.assertFalse(relay.observe(conn))
        self.assertEqual((conn.pending, conn.log.getvalue()), (7, ''))

    def test_rsp_seq_without_a_status_is_not_taken_for_a_refusal(self):
        device = Device(7, refused_status=0)
        relay, conn = relay_and_connection(device)
        self.assertFalse(relay.observe(conn))
        self.assertEqual((conn.pending, conn.anomalies), (7, 0))


if __name__ == '__main__':
    unittest.main()
