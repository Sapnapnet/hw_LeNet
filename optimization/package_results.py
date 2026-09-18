"""Package sources, data, final reports and synthesizable RTL, without tool caches."""
from pathlib import Path
import zipfile
import hashlib
import json

root=Path(__file__).resolve().parent.parent
assert (root/'optimization/REPORT_CN.md').exists(), 'Run make_report.py after all experiments complete'
destination=root.parent/'LeNet_optimized_verified.zip'
selected=[]
for p in root.rglob('*'):
    if not p.is_file(): continue
    rel=p.relative_to(root)
    if any(x in rel.parts for x in ('__pycache__','.git')): continue
    if rel.parts[0]=='build': continue
    if rel.parts[:2]==('optimization','build'):
        sub=rel.parts[3:]
        if len(sub)<3 or sub[0]!='solution1': continue
        if tuple(sub[1:3]) not in (('syn','verilog'),('syn','report'),('sim','report')): continue
    if p.suffix in ('.dcp','.wdb','.exe','.o','.dll','.jou'): continue
    if p.name.startswith('activity_recording.'): continue
    selected.append(p)
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in selected: z.write(p,'FPGA_LeNet_final/'+p.relative_to(root).as_posix())
    z.write(root.parent/'source_archive.json','source_archive.json')
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
sha=hashlib.sha256(destination.read_bytes()).hexdigest()
(destination.with_suffix('.sha256')).write_text(sha+'  '+destination.name+'\n',encoding='ascii')
print(json.dumps({'path':str(destination),'files':len(selected)+1,'bytes':destination.stat().st_size,'sha256':sha},indent=2))
