"""
uarm_ble.py
===========
Bluetooth LE driver for the uArm Swift Pro (no uFactory dongle, no USB).
Drop-in replacement for the parts of uarm.wrapper.SwiftAPI that
uarm_random_indentation.py uses:

    waiting_ready, set_mode, set_speed_factor, set_acceleration,
    set_report_position, set_position, flush_cmd, get_position,
    get_is_moving, disconnect

How it works:
    * Connects to the arm's BLE serial characteristic 0xFFE1 (service 0xFFE0) with bleak.
    * bleak is asyncio-based, so it runs in a background thread; every method here is
      a normal blocking call, like the SDK.
    * Commands are sent as "#n <gcode>\\n" and the reply "$n ..." is matched by n, so
      position reports or stale replies can never be mistaken for an answer. If the
      firmware does not answer with "$n", the driver falls back to plain commands
      (first "ok"/"E.." line is the answer).
    * BLE packets carry 20 bytes, so longer commands are split into chunks.
    * Waiting for the end of a move: polls M2200 (is moving?). If the firmware does not
      know M2200 (reply E20), it polls the position (P2220) until it stops changing.

Before use:
    * pip install bleak
    * uArm user-defined button DOWN (Bluetooth mode), then power on
    * Disconnect any phone / other program: the arm accepts one BLE connection at a time
    * Do NOT pair the arm in the Windows Bluetooth settings

Self-test (read-only, the arm does not move):
    python uarm_ble.py
"""

import asyncio
import queue
import re
import threading
import time

from bleak import BleakClient, BleakScanner

CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"   # 0xFFE1, found with nRF Connect
BLE_CHUNK = 20            # bytes per BLE write (default MTU 23 - 3)
CHUNK_GAP_S = 0.01        # pause between chunks of one command
REPLY_TIMEOUT_S = 3.0     # default wait for a command reply
WRITE_TIMEOUT_S = 5.0

_POS_RE = re.compile(r"X(-?\d+(?:\.\d+)?)\s+Y(-?\d+(?:\.\d+)?)\s+Z(-?\d+(?:\.\d+)?)")


class SwiftBLE:
    def __init__(self, address=None, name_prefix="UARM", scan_timeout=15.0, verbose=False):
        """address: BLE address (e.g. "7C:01:0A:5E:93:53"); if None or not found,
        the first device whose name starts with name_prefix is used."""
        self.verbose = verbose
        self.name = self.address = None
        self.firmware = None
        self._lines = queue.Queue()
        self._rx = bytearray()
        self._cmd_id = 0
        self._use_ids = True
        self._speed_factor = 1.0
        self._m2200 = None            # None = unknown, True/False = supported or not
        self._target = None
        self._lock = threading.Lock()
        self._client = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._run(self._connect(address, name_prefix, scan_timeout), scan_timeout + 30)
        print(f"[uArm BLE] connected to {self.name} ({self.address})")

    # ------------------------------------------------------------------ #
    # Background asyncio loop
    # ------------------------------------------------------------------ #
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro, timeout):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self, address, name_prefix, scan_timeout):
        dev = None
        if address:
            dev = await BleakScanner.find_device_by_address(address, timeout=scan_timeout)
        if dev is None:
            prefix = name_prefix.upper()
            dev = await BleakScanner.find_device_by_filter(
                lambda d, ad: (d.name or ad.local_name or "").upper().startswith(prefix),
                timeout=scan_timeout)
        if dev is None:
            raise RuntimeError("uArm not found over Bluetooth. Is the user button DOWN, the arm "
                               "powered on, and no phone/other program connected to it?")
        self._client = BleakClient(dev, disconnected_callback=self._on_disconnect)
        await self._client.connect()
        await self._client.start_notify(CHAR_UUID, self._on_notify)
        self.name, self.address = dev.name, dev.address

    def _on_disconnect(self, _client):
        print("[uArm BLE] WARNING: Bluetooth connection lost")

    def _on_notify(self, _sender, data):
        self._rx.extend(data)
        while b"\n" in self._rx:
            raw, _, rest = self._rx.partition(b"\n")
            self._rx = bytearray(rest)
            line = raw.decode(errors="ignore").strip()
            if line:
                if self.verbose:
                    print(f"[uArm BLE] <- {line}")
                self._lines.put(line)

    async def _write(self, data):
        for i in range(0, len(data), BLE_CHUNK):
            await self._client.write_gatt_char(CHAR_UUID, data[i:i + BLE_CHUNK], response=False)
            if i + BLE_CHUNK < len(data):
                await asyncio.sleep(CHUNK_GAP_S)

    # ------------------------------------------------------------------ #
    # Command / reply
    # ------------------------------------------------------------------ #
    def _drain(self):
        while True:
            try:
                self._lines.get_nowait()
            except queue.Empty:
                return

    def send_cmd_sync(self, cmd, timeout=REPLY_TIMEOUT_S):
        """Send one G-code line and return the reply text after the id / prefix,
        e.g. 'ok X169.40 Y-1.10 Z107.40' or 'E20'. Raises TimeoutError."""
        with self._lock:
            if self._use_ids:
                self._cmd_id = self._cmd_id % 9999 + 1
                n = self._cmd_id
                line_out = f"#{n} {cmd}\n"
            else:
                self._drain()
                line_out = f"{cmd}\n"
            if self.verbose:
                print(f"[uArm BLE] -> {line_out.strip()}")
            self._run(self._write(line_out.encode()), WRITE_TIMEOUT_S)

            t_end = time.time() + timeout
            while True:
                remaining = t_end - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"no reply to '{cmd}' within {timeout} s")
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError(f"no reply to '{cmd}' within {timeout} s") from None
                if self._use_ids:
                    if line == f"${n}" or line.startswith(f"${n} "):
                        return line[len(f"${n}"):].strip()
                elif line.startswith("ok") or re.match(r"^E\d", line):
                    return line
                # anything else: async report (@...) or a stale reply -> ignore

    @staticmethod
    def _is_error(reply):
        return bool(re.match(r"^E\d", reply))

    def _setting(self, cmd):
        """Non-critical setting: warn instead of failing if the firmware rejects it."""
        try:
            reply = self.send_cmd_sync(cmd)
        except TimeoutError as exc:
            print(f"[uArm BLE] WARNING: {exc}")
            return None
        if self._is_error(reply):
            print(f"[uArm BLE] WARNING: '{cmd}' -> {reply} (not supported by this firmware?)")
        return reply

    # ------------------------------------------------------------------ #
    # SwiftAPI-compatible methods
    # ------------------------------------------------------------------ #
    def waiting_ready(self, timeout=5):
        """Check the link (P2203). Falls back to commands without '#n' ids if needed."""
        try:
            reply = self.send_cmd_sync("P2203", timeout)
        except TimeoutError:
            print("[uArm BLE] no '$n' reply, switching to commands without ids")
            self._use_ids = False
            reply = self.send_cmd_sync("P2203", timeout)
        self.firmware = reply.replace("ok", "").strip()
        print(f"[uArm BLE] firmware {self.firmware}")
        return True

    def set_mode(self, mode=0):
        return self._setting(f"M2400 S{int(mode)}")

    def set_speed_factor(self, factor=1):
        """SDK-side only (multiplies F), exactly like SwiftAPI."""
        self._speed_factor = float(factor)

    def set_acceleration(self, acc):
        return self._setting(f"M204 P{acc} T{acc} R{acc}")

    def set_report_position(self, interval=0):
        return self._setting(f"M2120 V{interval}")

    def set_position(self, x, y, z, speed, wait=True, timeout=120, cmd="G0", **_ignored):
        """Move to (x, y, z) mm with F = speed * speed_factor. Raises on E.. replies."""
        f = speed * self._speed_factor
        gcode = f"{cmd} X{x:.2f} Y{y:.2f} Z{z:.2f} F{f:g}"
        self._target = (float(x), float(y), float(z))
        reply = self.send_cmd_sync(gcode, timeout if wait else REPLY_TIMEOUT_S)
        if self._is_error(reply):
            raise RuntimeError(f"move rejected: '{gcode}' -> {reply}")
        return reply

    def get_is_moving(self, timeout=REPLY_TIMEOUT_S):
        """True / False, or None if the firmware does not support M2200."""
        reply = self.send_cmd_sync("M2200", timeout)
        if self._is_error(reply):
            return None
        return "V1" in reply

    def get_position(self, wait=True, timeout=REPLY_TIMEOUT_S):
        """[x, y, z] in mm, or None."""
        try:
            reply = self.send_cmd_sync("P2220", timeout)
        except TimeoutError:
            return None
        m = _POS_RE.search(reply)
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None

    def flush_cmd(self, timeout=120, wait_stop=True):
        """Commands are synchronous already; with wait_stop, block until the arm stops."""
        if wait_stop:
            self._wait_stop(timeout)

    def _wait_stop(self, timeout):
        t0 = time.time()
        t_end = t0 + timeout
        time.sleep(0.05)

        if self._m2200 is not False:
            still = 0
            while time.time() < t_end:
                moving = self.get_is_moving()
                if moving is None:
                    self._m2200 = False
                    print("[uArm BLE] M2200 not supported, waiting on position instead")
                    break
                self._m2200 = True
                still = 0 if moving else still + 1
                if still >= 2 and time.time() - t0 >= 0.1:
                    return
                time.sleep(0.05)
            else:
                raise TimeoutError(f"arm still moving after {timeout} s")

        # Fallback: position stops changing (and is near the target, or 1 s has passed)
        prev, stable = None, 0
        while time.time() < t_end:
            p = self.get_position()
            if p is not None and prev is not None and max(abs(a - b) for a, b in zip(p, prev)) < 0.05:
                stable += 1
            else:
                stable = 0
            if p is not None:
                prev = p
            near = (self._target is None or
                    (p is not None and max(abs(a - b) for a, b in zip(p, self._target)) < 2.0))
            if stable >= 3 and (near or time.time() - t0 > 1.0):
                return
            time.sleep(0.1)
        raise TimeoutError(f"arm did not settle within {timeout} s")

    def disconnect(self):
        async def _close():
            try:
                await self._client.stop_notify(CHAR_UUID)
            except Exception:
                pass
            await self._client.disconnect()
        try:
            if self._client is not None:
                self._run(_close(), 10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            print("[uArm BLE] disconnected")


# --------------------------------------------------------------------------- #
# Self-test (read-only: nothing moves)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    arm = SwiftBLE(verbose=True)
    try:
        arm.waiting_ready()
        print("position :", arm.get_position())
        moving = arm.get_is_moving()
        print("M2200    :", "not supported (position fallback will be used)" if moving is None
              else f"supported, moving = {moving}")
        arm.set_report_position(interval=0)
        print("Self-test OK")
    finally:
        arm.disconnect()