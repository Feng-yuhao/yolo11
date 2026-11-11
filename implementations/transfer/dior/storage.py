"""Preserve all predictions losslessly without an intermediate giant plain JSON."""
import gzip
import json
import os
import shutil
from common import require

RESERVE=1024**3
REPORT_CAP=int(3.5*1024**3)
def disk_guard(path):
    free=shutil.disk_usage(path).free;require(free>=RESERVE,'Less than1GiB free; stop safely, retain partial output, never delete data')
    return free
class PredictionWriter:
    def __init__(self,path,report):
        self.path=path;self.count=0;disk_guard(path.parent)
        self.other_bytes=sum(p.stat().st_size for p in report.rglob('*') if p.is_file())
        self.raw=path.open('xb');self.stream=gzip.GzipFile(filename='',fileobj=self.raw,mode='wb',compresslevel=6,mtime=0)
        self.stream.write(b'[');self.closed=False
    def add(self,rows):
        disk_guard(self.path.parent)
        for row in rows:
            if self.count:self.stream.write(b',')
            self.stream.write(json.dumps(row,separators=(',',':'),allow_nan=False).encode('utf-8'));self.count+=1
        self.stream.flush()
        require(self.raw.tell()<512*1024**2 and self.other_bytes+self.raw.tell()<REPORT_CAP,'Compressed prediction budget exceeded; partial file retained')
    def close(self):
        if not self.closed:
            self.stream.write(b']\n');self.stream.close();self.raw.flush();os.fsync(self.raw.fileno());self.raw.close();self.closed=True
    def abort(self):
        if not self.closed:self.stream.close();self.raw.close();self.closed=True
def read_predictions(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
