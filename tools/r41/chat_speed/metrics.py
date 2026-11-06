"""Fail closed when board telemetry cannot establish device-only execution."""


def device_only(turn):
    return (turn.get('cpu_numerical_fallback') is False
            and turn.get('steady_decode_host_reads_detected') is False)
