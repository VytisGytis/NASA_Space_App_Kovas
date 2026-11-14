"""
Capture side of the pipeline: ESP32 laser + Open Camera over wireless ADB.

Same flow as `android to computer image transfer/volumedownactivation.py`
(laser on -> one volume-down press -> laser off once N photos exist -> pull
-> verify -> delete from phone -> done.txt), but driven by a config dict and
a status callback so the GUI can use it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

import requests

DEFAULTS = {
    "esp_ip": "",
    "adb": shutil.which("adb") or "adb",
    "phone": "",                         # "ip:port" from Wireless debugging, or ""
    "remote": "/sdcard/DCIM/OpenCamera",
    "n_photos": 3,                       # must match Open Camera's repeat count
    "settle": 1.0,                       # laser warm-up before the shutter press
    "max_laser_s": 8.0,                  # keep < ESP32 failsafe (10 s)
    "write_timeout": 25.0,
    "delete_after_pull": True,
}


class Capture:
    def __init__(self, cfg: dict, incoming: str, status=print):
        self.cfg = {**DEFAULTS, **cfg}
        self.incoming = incoming
        self.status = status
        self.pending: dict[str, int] = {}
        self.pending_folder: str | None = None

    # ---------- adb ----------
    def adb(self, *args, timeout=120):
        return subprocess.run([self.cfg["adb"], *args], capture_output=True,
                              text=True, timeout=timeout)

    def phone_online(self) -> bool:
        try:
            lines = self.adb("devices", timeout=10).stdout.splitlines()[1:]
        except (OSError, subprocess.SubprocessError):
            return False
        return any(l.strip().endswith("\tdevice") for l in lines)

    def ensure_connected(self) -> bool:
        if self.phone_online():
            return True
        if self.cfg["phone"]:
            try:
                self.adb("connect", self.cfg["phone"], timeout=15)
            except (OSError, subprocess.SubprocessError):
                return False
            time.sleep(1)
            return self.phone_online()
        return False

    def remote_dngs(self) -> dict[str, int]:
        out = self.adb("shell", f"stat -c '%n|%s' {self.cfg['remote']}/*.dng 2>/dev/null").stdout
        files = {}
        for line in out.splitlines():
            line = line.strip()
            if "|" in line:
                path, size = line.rsplit("|", 1)
                if size.isdigit():
                    files[path] = int(size)
        return files

    def _new(self, before):
        return {p: s for p, s in self.remote_dngs().items() if p not in before}

    # ---------- laser ----------
    def laser(self, on: bool):
        requests.get(f"http://{self.cfg['esp_ip']}/{'on' if on else 'off'}", timeout=3)

    # ---------- scan ----------
    def scan(self) -> str | None:
        """Run one full scan. Returns the new folder path, or None on failure."""
        c = self.cfg
        if not c["esp_ip"]:
            self.status("Enter the ESP32 IP first")
            return None
        if not shutil.which(c["adb"]) and not os.path.exists(c["adb"]):
            self.status(f"adb not found ('{c['adb']}'). Set its path in Settings.")
            return None

        self.status("Checking phone and laser...")
        if not self.ensure_connected():
            self.status("No phone on adb (check Wireless debugging / phone ip:port)")
            return None
        try:
            self.laser(False)   # harmless ping
        except requests.RequestException:
            self.status("ESP32 not reachable - check the IP and the hotspot")
            return None

        before = self.remote_dngs()
        seen = 0
        try:
            self.status("Laser on...")
            self.laser(True)
            time.sleep(c["settle"])
            self.status("Shutter pressed, Open Camera is capturing...")
            self.adb("shell", "input keyevent KEYCODE_VOLUME_DOWN")
            t0 = time.time()
            while time.time() - t0 < c["max_laser_s"]:
                seen = len(self._new(before))
                if seen >= c["n_photos"]:
                    break
                time.sleep(0.3)
        finally:
            try:
                self.laser(False)
            except requests.RequestException:
                self.status("WARNING: laser OFF failed (ESP32 failsafe will cut it)")

        self.status(f"Laser off. {seen} photo(s) seen, waiting for files to finish...")
        new = self._wait_stable(before)
        if not new:
            self.status("No new DNGs. Is Open Camera in the foreground with RAW on?")
            return None

        folder = os.path.join(self.incoming, time.strftime("scan_%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        ok, failed = self._pull(new, folder)
        return self._finish(folder, ok, failed, c["n_photos"])

    def retry(self) -> str | None:
        if not self.pending:
            return None
        if not self.ensure_connected():
            self.status("No phone on adb")
            return None
        ok, failed = self._pull(self.pending, self.pending_folder)
        return self._finish(self.pending_folder, ok, failed, None, append=True)

    def _wait_stable(self, before):
        t0, prev = time.time(), {}
        while time.time() - t0 < self.cfg["write_timeout"]:
            cur = self._new(before)
            if cur and cur == prev and all(s > 0 for s in cur.values()):
                return cur
            prev = cur
            time.sleep(0.7)
        return prev

    def _pull(self, files, folder):
        ok, failed = [], {}
        for path, size in sorted(files.items()):
            name = os.path.basename(path)
            self.status(f"Pulling {name}...")
            dest = os.path.join(folder, name)
            try:
                r = self.adb("pull", path, dest)
                if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) == size:
                    ok.append(name)
                    if self.cfg["delete_after_pull"]:
                        self.adb("shell", f"rm '{path}'")
                else:
                    failed[path] = size
            except (OSError, subprocess.SubprocessError):
                failed[path] = size
        return ok, failed

    def _finish(self, folder, ok, failed, expected, append=False):
        self.pending = failed
        self.pending_folder = folder if failed else None
        if not failed:
            with open(os.path.join(folder, "done.txt"), "a" if append else "w") as f:
                f.write("\n".join(ok) + "\n")
        name = os.path.basename(folder)
        if failed:
            self.status(f"{len(ok)} pulled, {len(failed)} FAILED (still on phone) - press Retry")
        elif expected and len(ok) < expected:
            self.status(f"Only {len(ok)}/{expected} photos arrived -> {name}")
        else:
            self.status(f"Done: {len(ok)} photos -> {name}")
        return folder


def detect_esp_ip(timeout=15) -> str | None:
    """Read 'Connected. IP: x.x.x.x' from the ESP32 over USB serial
    (the Arduino serial monitor must be closed)."""
    try:
        import serial
        import serial.tools.list_ports
    except ImportError:
        return None
    pattern = re.compile(r"IP:\s*(\d+\.\d+\.\d+\.\d+)")
    for port in serial.tools.list_ports.comports():
        try:
            with serial.Serial(port.device, 115200, timeout=0.5) as s:
                s.dtr = False
                s.rts = True
                time.sleep(0.1)
                s.rts = False
                t0 = time.time()
                while time.time() - t0 < timeout:
                    m = pattern.search(s.readline().decode(errors="ignore"))
                    if m:
                        return m.group(1)
        except Exception:  # noqa: BLE001 - busy / not an ESP32, try the next port
            continue
    return None
