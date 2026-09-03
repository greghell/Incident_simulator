#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transparent srsRAN 4G downlink broker with repeated labeled captures.

Keep this broker running between srsENB (:2000) and srsUE (:2100).  At the
interactive prompt, type one of:

    idle
    light
    medium
    loaded

and the next capture window is written as ``srsran_<state>.cf32``.  The broker
continues forwarding samples after every capture, so the LTE session stays up.
"""
from __future__ import annotations

import argparse
import json
import signal as pysignal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gnuradio import gr, zeromq


class RepeatedComplex64Capture(gr.sync_block):
    def __init__(self, sample_rate_hz: float, capture_seconds: float):
        gr.sync_block.__init__(
            self,
            name="repeated_complex64_capture",
            in_sig=[np.complex64],
            out_sig=None,
        )
        self.sample_rate_hz = float(sample_rate_hz)
        self.capture_seconds = float(capture_seconds)
        self.n_capture = int(round(self.sample_rate_hz * self.capture_seconds))
        if self.n_capture <= 0:
            raise ValueError("capture_seconds must produce at least one sample")
        self._trigger = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._fh = None
        self._written = 0
        self._output_path = None

    @property
    def done_event(self):
        return self._done

    @property
    def output_path(self):
        return self._output_path

    def trigger(self, output_path: Path):
        with self._lock:
            if self._fh is not None or self._trigger.is_set():
                raise RuntimeError("a capture is already running")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path = output_path
            self._fh = open(output_path, "wb")
            self._written = 0
            self._done.clear()
            self._trigger.set()

    def stop(self):
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
        return True

    def work(self, input_items, output_items):
        x = input_items[0]
        if not self._trigger.is_set():
            return len(x)
        with self._lock:
            if self._fh is None:
                return len(x)
            remaining = self.n_capture - self._written
            n = min(len(x), remaining)
            if n > 0:
                np.asarray(x[:n], dtype=np.complex64).tofile(self._fh)
                self._written += n
            if self._written >= self.n_capture:
                self._fh.flush()
                self._fh.close()
                self._fh = None
                self._trigger.clear()
                self._done.set()
        return len(x)


class SrsranDownlinkBroker(gr.top_block):
    def __init__(self, enb_tx_endpoint: str, ue_rx_endpoint: str,
                 sample_rate_hz: float, capture_seconds: float):
        super().__init__("srsRAN LTE downlink repeated-capture broker")
        itemsize = gr.sizeof_gr_complex
        self.enb_source = zeromq.req_source(itemsize, 1, enb_tx_endpoint, 100, False, -1)
        self.ue_sink = zeromq.rep_sink(itemsize, 1, ue_rx_endpoint, 100, False, -1)
        self.capture_sink = RepeatedComplex64Capture(sample_rate_hz, capture_seconds)
        self.connect(self.enb_source, self.ue_sink)
        self.connect(self.enb_source, self.capture_sink)


def write_sidecar(path: Path, args, state: str):
    expected_samples = int(round(args.sample_rate * args.capture_seconds))
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps({
        "capture_file": path.name,
        "state": state,
        "dtype": "complex64",
        "sample_rate_hz": args.sample_rate,
        "capture_seconds": args.capture_seconds,
        "expected_samples": expected_samples,
        "expected_bytes": expected_samples * np.dtype(np.complex64).itemsize,
        "capture_completed_utc": datetime.now(timezone.utc).isoformat(),
        "enb_tx_endpoint": args.enb_tx,
        "ue_rx_endpoint": args.ue_rx,
        "note": "Genuine srsRAN 4G eNodeB downlink captured by transparent GNU Radio broker; load label supplied interactively by operator.",
    }, indent=2), encoding="utf-8")
    return sidecar


def parse_args():
    p = argparse.ArgumentParser(description="Repeated labeled srsRAN downlink IQ capture")
    p.add_argument("--enb-tx", default="tcp://127.0.0.1:2000")
    p.add_argument("--ue-rx", default="tcp://*:2100")
    p.add_argument("--sample-rate", type=float, default=11.52e6)
    p.add_argument("--capture-seconds", type=float, default=0.50)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tb = SrsranDownlinkBroker(args.enb_tx, args.ue_rx, args.sample_rate, args.capture_seconds)
    stopping = threading.Event()

    def handle_signal(signum, frame):
        stopping.set()

    pysignal.signal(pysignal.SIGINT, handle_signal)
    pysignal.signal(pysignal.SIGTERM, handle_signal)

    print("srsRAN 4G GNU Radio multi-capture broker")
    print("==========================================")
    print(f"eNB TX source: {args.enb_tx}")
    print(f"UE RX sink:    {args.ue_rx}")
    print(f"Sample rate:   {args.sample_rate/1e6:.3f} Msps")
    print(f"Capture:       {args.capture_seconds:.3f} s each")
    print(f"Output dir:    {args.output_dir.resolve()}")
    print()
    print("Commands: idle, light, medium, loaded, q")
    print("Change traffic in another terminal BEFORE entering the matching label here.")

    tb.start()
    try:
        while not stopping.is_set():
            try:
                cmd = input("\nCapture state [idle/light/medium/loaded/q]: ").strip().lower()
            except EOFError:
                break
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd not in {"idle", "light", "medium", "loaded"}:
                print("Unknown command. Use idle, light, medium, loaded, or q.")
                continue
            out = args.output_dir / f"srsran_{cmd}.cf32"
            if out.exists():
                ans = input(f"{out.name} exists. Overwrite? [y/N]: ").strip().lower()
                if ans not in {"y", "yes"}:
                    continue
            tb.capture_sink.trigger(out)
            print(f"Capturing {cmd!r} state -> {out} ...")
            while not tb.capture_sink.done_event.wait(timeout=0.2):
                if stopping.is_set():
                    break
            if not tb.capture_sink.done_event.is_set():
                break
            sidecar = write_sidecar(out, args, cmd)
            expected_samples = int(round(args.sample_rate * args.capture_seconds))
            expected_bytes = expected_samples * np.dtype(np.complex64).itemsize
            print(f"Complete: {out.name} ({expected_bytes/1e6:.1f} MB)")
            print(f"Metadata: {sidecar.name}")
            print("Broker remains live; change traffic and capture another state when ready.")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping GNU Radio broker...")
        tb.stop()
        tb.wait()
        print("Stopped.")


if __name__ == "__main__":
    main()
