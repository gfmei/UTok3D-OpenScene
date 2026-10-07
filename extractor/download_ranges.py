"""Resume a public HTTP download in verified byte ranges; retain progress metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import threading
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    with urlopen(Request(a.url, method='HEAD'), timeout=30) as r:
        size = int(r.headers['Content-Length'])
    chunk = 64 * 1024**2
    sidecar = a.output.with_suffix(a.output.suffix + '.progress.json')
    if sidecar.exists():
        state = json.loads(sidecar.read_text())
        if (state['url'], state['bytes'], state['chunk']) != (a.url, size, chunk):
            raise ValueError('Download metadata mismatch')
    else:
        prefix = a.output.stat().st_size if a.output.exists() else 0
        state = dict(url=a.url, bytes=size, chunk=chunk,
                     completed=list(range(prefix // chunk)))
    fd = os.open(a.output, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)
    done = set(state['completed'])
    total = (size + chunk - 1) // chunk
    write_lock = threading.Lock()

    def fetch(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        for attempt in range(6):
            try:
                req = Request(a.url, headers={'Range': f'bytes={start}-{end}'})
                with urlopen(req, timeout=90) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server did not honor byte range')
                    data = r.read(end-start+1)
                    if len(data) != end-start+1:
                        raise IOError('Incomplete range')
                    # One contiguous write per range avoids interleaved 1 MB
                    # writes from many threads on the spinning data disk.
                    with write_lock:
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], start + written)
                return i
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 16))

    started = time.monotonic()
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(total) if i not in done]
        for f in as_completed(futures):
            done.add(f.result())
            state['completed'] = sorted(done)
            tmp = sidecar.with_suffix('.tmp')
            tmp.write_text(json.dumps(state))
            tmp.replace(sidecar)
            if len(done) % 8 == 0 or len(done) == total:
                print(f'{a.output.name}: {len(done)}/{total} ranges; '
                      f'{100*len(done)/total:.1f}%; elapsed {time.monotonic()-started:.0f}s', flush=True)
    os.fsync(fd)
    os.close(fd)
    print('DOWNLOAD COMPLETE', a.output, flush=True)


if __name__ == '__main__':
    main()
