"""Read-only ownership check for this CANNLab backend. Never starts/stops models."""
import os
import json
import urllib.request
from pathlib import Path


class ProtectionError(RuntimeError):
    pass


class AscendProtection:
    def __init__(self, profile):
        self.root = Path(profile["root"])
        self.devices = profile.get("npu_device_paths", ["/dev/davinci10", "/dev/davinci11"])
        self.npu_count = int(profile.get("npu_count", 2))
        self.worker_health_url = profile.get("worker_health_url")
        self.maintenance_file = profile.get("maintenance_file")

    def check(self):
        if self.maintenance_file and Path(self.maintenance_file).exists():
            raise ProtectionError("Recovery maintenance in progress; no new task accepted")
        try:
            pid = int((self.root/"logs/comfyui.pid").read_text().strip())
            proc = Path("/proc")/str(pid)
            uid = proc.stat().st_uid
            cwd = (proc/"cwd").resolve(strict=True)
            argv = (proc/"cmdline").read_bytes().split(b"\0")
            if uid != os.getuid() or cwd != (self.root/"ComfyUI").resolve() or b"main.py" not in argv:
                raise ProtectionError("Backend process identity does not match the owned CANNLab deployment")
            devices = self.devices
            if len(devices) != self.npu_count or len(set(devices)) != len(devices) or not all(Path(p).exists() for p in devices):
                raise ProtectionError("The configured pair of NPU devices is unavailable")
        except (OSError, ValueError) as error:
            raise ProtectionError("Owned ComfyUI process is unavailable") from error
        if self.worker_health_url:
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(self.worker_health_url, timeout=5) as response:
                    state = json.load(response)
                if state.get("worker_alive") is not True:
                    raise ProtectionError("ComfyUI generation worker is not alive; no new task accepted")
            except (OSError, ValueError) as error:
                raise ProtectionError("ComfyUI generation worker health is unavailable; no new task accepted") from error
        return {"policy": "owned_cannlab_instance", "backend_pid": pid, "npu_count": self.npu_count,
                "devices": devices, "read_only_guard": True, "thermal_power_protection": False}
