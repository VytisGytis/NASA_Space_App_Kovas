"""
Laser speckle scan pipeline.

SCAN button -> laser ON -> ONE volume-down press (Open Camera does the repeat)
            -> laser OFF once N photos exist -> pull new DNGs into incoming/scan_<time>/
            -> verify size -> delete from phone -> done.txt marker

Open Camera must be set up for the repeat itself (photo count + interval) and
volume keys must be set to "Take photo".

Requirements:  pip install requests pyserial
               adb (platform-tools) on PATH
"""
import os
import re
import subprocess
import threading
import time
import tkinter as tk

import requests

# ---------------- CONFIG ----------------
ADB = r"C:\platform-tools\adb.exe"                          # or full path, e.g. r"C:\platform-tools\adb.exe"
PHONE = ""                           # "ip:port" from Wireless debugging, or "" to skip auto-connect
REMOTE = "/sdcard/DCIM/OpenCamera"   # Open Camera save folder
LOCAL = "incoming"                   # folder your team reads from
N_PHOTOS = 3                         # must match the repeat count set in Open Camera
SETTLE = 1                         # laser warm-up before the shutter press
MAX_LASER_S = 8                      # max seconds the laser stays on (keep < ESP32 failsafe of 10 s)
WRITE_TIMEOUT = 25                   # max seconds to wait for files to finish writing
DELETE_AFTER_PULL = True             # flush phone after a verified pull
# ----------------------------------------

os.makedirs(LOCAL, exist_ok=True)


def adb(*args, timeout=120):
    return subprocess.run([ADB] + list(args), capture_output=True, text=True, timeout=timeout)


def ensure_connected():
    """True if an adb device is online (tries to connect to PHONE if set)."""
    def online():
        lines = adb("devices").stdout.splitlines()[1:]
        return any(l.strip().endswith("\tdevice") for l in lines)

    if online():
        return True
    if PHONE:
        adb("connect", PHONE)
        time.sleep(1)
        return online()
    return False


def remote_dngs():
    """Return {remote_path: size_bytes} for every DNG in the Open Camera folder."""
    out = adb("shell", f"stat -c '%n|%s' {REMOTE}/*.dng 2>/dev/null").stdout
    files = {}
    for line in out.splitlines():
        line = line.strip()
        if "|" in line:
            path, size = line.rsplit("|", 1)
            if size.isdigit():
                files[path] = int(size)
    return files


def laser(esp_ip, state):
    requests.get(f"http://{esp_ip}/{'on' if state else 'off'}", timeout=3)


def new_files(before):
    return {p: s for p, s in remote_dngs().items() if p not in before}


def wait_for_count(before, n, timeout):
    """Wait until at least n new DNGs exist (or timeout). Returns count seen."""
    t0 = time.time()
    count = 0
    while time.time() - t0 < timeout:
        count = len(new_files(before))
        if count >= n:
            break
        time.sleep(0.3)
    return count


def wait_for_stable(before, timeout):
    """Wait until the new files' sizes stop changing (fully written)."""
    t0 = time.time()
    prev = {}
    while time.time() - t0 < timeout:
        cur = new_files(before)
        if cur and cur == prev and all(s > 0 for s in cur.values()):
            return cur
        prev = cur
        time.sleep(0.7)
    return prev


def pull_and_flush(files, folder, status):
    """Pull each file, verify size, delete from phone. Returns (ok_list, failed_dict)."""
    ok, failed = [], {}
    for path, size in sorted(files.items()):
        name = os.path.basename(path)
        status(f"Pulling {name}...")
        dest = os.path.join(folder, name)
        try:
            r = adb("pull", path, dest)
            if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) == size:
                ok.append(name)
                if DELETE_AFTER_PULL:
                    adb("shell", f"rm '{path}'")
            else:
                failed[path] = size
        except Exception:
            failed[path] = size
    return ok, failed


def detect_esp_ip(timeout=15):
    """
    Read the ESP32's IP from its serial output ("Connected. IP: x.x.x.x").
    Tries every serial port. The Arduino serial monitor must be closed
    (only one program can use the port). Opening the port usually resets the
    ESP32, so it prints its IP again after reconnecting to Wi-Fi.
    """
    import serial
    import serial.tools.list_ports

    pattern = re.compile(r"IP:\s*(\d+\.\d+\.\d+\.\d+)")
    for port in serial.tools.list_ports.comports():
        try:
            with serial.Serial(port.device, 115200, timeout=0.5) as s:
                # pulse the reset line so the board reboots and prints its IP
                s.dtr = False
                s.rts = True
                time.sleep(0.1)
                s.rts = False
                t0 = time.time()
                while time.time() - t0 < timeout:
                    line = s.readline().decode(errors="ignore")
                    m = pattern.search(line)
                    if m:
                        return m.group(1)
        except Exception:
            continue  # port busy or not an ESP32, try the next one
    return None


class App:
    def __init__(self):
        self.pending = {}
        self.pending_folder = None

        self.root = tk.Tk()
        self.root.title("Laser scan")
        self.msg = tk.StringVar(value="Enter the ESP32 IP (from the Arduino serial monitor)")

        row = tk.Frame(self.root)
        row.pack(padx=40, pady=(25, 5))
        tk.Label(row, text="ESP32 IP:", font=("Arial", 12)).pack(side="left")
        self.ip_var = tk.StringVar()
        tk.Entry(row, textvariable=self.ip_var, width=16, font=("Arial", 12)).pack(side="left", padx=6)
        tk.Button(row, text="Detect", command=self.detect).pack(side="left")

        self.btn = tk.Button(self.root, text="SCAN", font=("Arial", 28), width=12,
                             command=lambda: self.start(self.scan))
        self.btn.pack(padx=40, pady=(10, 10))
        self.retry_btn = tk.Button(self.root, text="Retry failed pulls", state="disabled",
                                   command=lambda: self.start(self.retry))
        self.retry_btn.pack(pady=(0, 10))
        tk.Label(self.root, textvariable=self.msg, font=("Arial", 12),
                 wraplength=420).pack(padx=20, pady=(0, 20))

    # ---- UI helpers (thread-safe) ----
    def status(self, s):
        self.root.after(0, self.msg.set, s)
        print(s)

    def set_busy(self, busy):
        self.root.after(0, lambda: self.btn.config(state="disabled" if busy else "normal"))
        self.root.after(0, lambda: self.retry_btn.config(
            state="normal" if (self.pending and not busy) else "disabled"))

    def detect(self):
        def work():
            self.status("Reading ESP32 IP from serial (close the Arduino serial monitor)...")
            try:
                ip = detect_esp_ip()
            except Exception as e:
                ip = None
                print(e)
            if ip:
                self.root.after(0, self.ip_var.set, ip)
                self.status(f"ESP32 found at {ip}")
            else:
                self.status("Could not detect the IP. Is the ESP32 on USB and the "
                            "serial monitor closed? You can type the IP instead.")
        threading.Thread(target=work, daemon=True).start()

    def start(self, fn):
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn):
        self.set_busy(True)
        try:
            fn()
        except Exception as e:
            self.status(f"Error: {e}")
        finally:
            self.set_busy(False)

    # ---- main scan ----
    def scan(self):
        esp = self.ip_var.get().strip()
        if not esp:
            self.status("Type the ESP32 IP first")
            return

        # pre-checks
        self.status("Checking phone and laser...")
        if not ensure_connected():
            self.status("adb: no phone connected (check Wireless debugging / adb connect)")
            return
        try:
            laser(esp, False)  # harmless ping, confirms the ESP32 is reachable
        except Exception:
            self.status("ESP32 not reachable - check the IP and the hotspot")
            return

        before = remote_dngs()

        # laser on -> one shutter press -> laser off when the photos exist
        try:
            self.status("Laser on...")
            laser(esp, True)
            time.sleep(SETTLE)

            self.status("Shutter pressed, Open Camera is capturing...")
            adb("shell", "input keyevent KEYCODE_VOLUME_DOWN")
            seen = wait_for_count(before, N_PHOTOS, MAX_LASER_S)
        finally:
            try:
                laser(esp, False)
            except Exception:
                self.status("WARNING: could not send laser OFF (ESP32 failsafe will cut it)")

        # wait for files to be fully written
        self.status(f"Laser off. {seen} photo(s) seen, waiting for files to finish writing...")
        new = wait_for_stable(before, WRITE_TIMEOUT)
        if not new:
            self.status("No new DNGs appeared. Is Open Camera in the foreground with RAW on?")
            return

        # pull, verify, flush
        folder = os.path.join(LOCAL, time.strftime("scan_%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        ok, failed = pull_and_flush(new, folder, self.status)
        self.finish(folder, ok, failed, expected=N_PHOTOS)

    def retry(self):
        if not self.pending:
            return
        if not ensure_connected():
            self.status("adb: no phone connected")
            return
        ok, failed = pull_and_flush(self.pending, self.pending_folder, self.status)
        self.finish(self.pending_folder, ok, failed, expected=None, append=True)

    def finish(self, folder, ok, failed, expected, append=False):
        self.pending = failed
        self.pending_folder = folder if failed else None

        if not failed:
            with open(os.path.join(folder, "done.txt"), "a" if append else "w") as f:
                f.write("\n".join(ok) + "\n")

        if failed:
            self.status(f"{len(ok)} pulled, {len(failed)} FAILED (still on phone). "
                        f"Press Retry. Folder: {folder}")
        elif expected and len(ok) < expected:
            self.status(f"Only {len(ok)}/{expected} photos arrived -> {folder}")
        else:
            self.status(f"Done: {len(ok)} photos -> {folder}")

    def run(self):
        self.root.after(300, self.detect)   # try auto-detect on startup
        self.root.mainloop()


if __name__ == "__main__":
    App().run()