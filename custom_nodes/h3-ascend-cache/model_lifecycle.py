"""One resident text encoder family, shared by the serial prompt worker.

This cache holds model weights only. Conditioning, media and prompt results keep
their existing persistent cache identities and are never reused through here.
"""
import gc
import json
import logging
import os
from pathlib import Path
import threading
import time


class SingleModelCache:
    def __init__(self):
        self._lock = threading.RLock()
        self._key = None
        self._value = None

    def get(self, key, factory, release):
        with self._lock:
            if self._value is not None and self._key == key:
                return self._value, True
            if self._value is not None:
                # Do not allocate a replacement until the old family is offloaded.
                release(self._value)
                self._value = None
                self._key = None
            gc.collect()
            value = factory()
            if value is None:
                raise ValueError('Text encoder factory returned no model')
            self._key, self._value = key, value
            return value, False


text_encoder_cache = SingleModelCache()


def record_memory(stage):
    """Observe this service's memory without reading other processes' arguments."""
    try:
        cgroup = Path('/sys/fs/cgroup/memory')
        status = Path('/proc/self/status').read_text()
        rss = next(int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:'))
        event = {'unix':time.time(),'pid':os.getpid(),'stage':stage,'rss_bytes':rss}
        for name in ['memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes']:
            path = cgroup/name
            if path.exists():event[name]=int(path.read_text())
        path = Path('/mnt/workspace/h3-ascend/logs/model-memory.jsonl')
        with path.open('a') as output:output.write(json.dumps(event)+'\n')
        logging.info('H3 memory %s rss=%.2fGiB cgroup=%.2fGiB',stage,rss/2**30,
                     event.get('memory.usage_in_bytes',0)/2**30)
    except (OSError,ValueError,StopIteration):
        logging.warning('H3 memory telemetry unavailable',exc_info=True)
