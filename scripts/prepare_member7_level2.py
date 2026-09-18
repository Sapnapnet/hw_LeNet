"""Incremental host execution of the unchanged HLS C++ preprocessing core.
Requires Python 3.9+, Windows PowerShell/System.Drawing and MinGW g++.
No model inference, HLS synthesis, or image algorithm is implemented here.
"""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,datetime,sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/self_collected'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def local(path):
 p=(DATA/path).resolve()
 if not p.is_relative_to(DATA.resolve()):raise ValueError('Path outside data directory: '+path)
 return p
def pgm(p):
 b=p.read_bytes();header=b'P5\n28 28\n255\n'
 if not b.startswith(header) or len(b)!=len(header)+784:raise ValueError('Invalid PGM: '+str(p))
 return b[len(header):]
def verify(row):
 for mode in ['simple','full']:
  vals=pgm(local(row[mode+'_relative']))
  nums=[int(x) for x in local(row['fixed_'+mode+'_relative']).read_text().splitlines()]
  if nums!=[(v*32+127)//255 for v in vals]:raise ValueError('Fixed encoding mismatch: '+row['filename'])
  if mode=='full' and set(vals)-{0,255}:raise ValueError('Nonbinary Full')
 if not local('original/'+row['source_relative']).is_file():raise ValueError('Missing original')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--compiler',default='g++');args=ap.parse_args()
 if not args.batch or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.batch):raise ValueError('Use ASCII batch identifier')
 rows=readcsv(DATA/'label.csv');ids=[r['sample_id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample ID')
 sources=json.loads((DATA/'source_manifest.json').read_text(encoding='utf8'))
 provenance={r['sample_id']:r for r in sources['samples']}
 inputs={}
 for r in rows:
  if not r['label'].isdigit() or not 0<=int(r['label'])<=9:raise ValueError('Digit labels only')
  if r['sample_id']!=Path(r['filename']).stem or r['original_path']!='original/'+r['label']+'/'+r['filename']:raise ValueError('Identity/path mismatch')
  h=sha(local(r['original_path']))
  if h!=provenance[r['sample_id']]['sha256']:raise ValueError('Source manifest hash mismatch')
  inputs[r['sample_id']]=h
 codefiles=['hls/preprocess/level2_preprocess.cpp','hls/preprocess/level2_preprocess.h','scripts/level2_preprocess_driver.cpp','scripts/export_level2_rgb.ps1','scripts/prepare_member7_level2.py']
 code={p:sha(ROOT/p) for p in codefiles};pipeline=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
 statepath=DATA/'processing_state.json';state=json.loads(statepath.read_text(encoding='utf8')) if statepath.exists() else {'samples':{}}
 pending=[r for r in rows if state['samples'].get(r['sample_id'],{}).get('input_sha256')!=inputs[r['sample_id']] or state['samples'].get(r['sample_id'],{}).get('pipeline_sha256')!=pipeline]
 previous=readcsv(DATA/'level2_manifest.csv') if (DATA/'level2_manifest.csv').exists() else []
 byid={r['sample_id']:r for r in previous}
 for r in rows:
  if r not in pending:
   if r['sample_id'] not in byid:raise ValueError('Missing cumulative manifest record')
   verify(byid[r['sample_id']])
 if not pending:print('No new or changed samples; existing outputs verified.');return
 batch=DATA/'batches'/args.batch
 if batch.exists():raise ValueError('Batch exists; use a new identifier (failed batches are retained)')
 batch.mkdir(parents=True);(batch/'raw_rgb').mkdir()
 adapter=[{'filename':r['filename'],'label':r['label'],'writer_id':'m'+r['member'],'folder':r['label']} for r in rows]
 writecsv(DATA/'original/label.csv',adapter,['filename','label','writer_id','folder'])
 pendingids={r['sample_id'] for r in pending};writecsv(batch/'input_labels.csv',[r for r in adapter if Path(r['filename']).stem in pendingids],['filename','label','writer_id','folder'])
 def run(cmd):
  with (batch/'run.log').open('a',encoding='utf8') as log:
   log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush()
   subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/export_level2_rgb.ps1'),'-OriginalRoot',str(DATA/'original'),'-RawRgbRoot',str(batch/'raw_rgb'),'-LabelCsvPath',str(batch/'input_labels.csv')])
 binary=ROOT/'build/member8_level2/level2_preprocess_driver.exe';binary.parent.mkdir(parents=True,exist_ok=True)
 run([args.compiler,'-std=c++11','-O2',str(ROOT/codefiles[0]),str(ROOT/codefiles[2]),'-o',str(binary)])
 run([str(binary),str(batch/'raw_rgb'),str(batch/'raw_rgb/manifest.csv'),str(batch)])
 generated=readcsv(batch/'level2_manifest.csv')
 if len(generated)!=len(pending):raise ValueError('Output count mismatch')
 base=batch.relative_to(DATA).as_posix()
 for r in generated:
  sid=Path(r['filename']).stem
  if sid not in pendingids:raise ValueError('Unexpected sample')
  r={'sample_id':sid,**r,'batch':args.batch,'input_sha256':inputs[sid],'pipeline_sha256':pipeline}
  for key in ['simple_relative','full_relative','fixed_simple_relative','fixed_full_relative']:r[key]=base+'/'+r[key].replace('\\','/')
  verify(r);byid[sid]=r
  state['samples'][sid]={'input_sha256':inputs[sid],'pipeline_sha256':pipeline,'batch':args.batch}
 cumulative=[byid[r['sample_id']] for r in rows]
 for r in cumulative:verify(r)
 # Publish a single index only after the complete batch has passed verification.
 writecsv(DATA/'level2_manifest.next.csv',cumulative,list(cumulative[0]));(DATA/'level2_manifest.next.csv').replace(DATA/'level2_manifest.csv')
 state['samples']={sid:state['samples'][sid] for sid in ids};dump(statepath,state)
 report={'status':'passed','batch':args.batch,'processed':len(pending),'cumulative':len(rows),'pipeline_sha256':pipeline,'code_sha256':code,'source_dataset_version':sources['batch'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_roi_invalid':[r['sample_id'] for r in cumulative if r['full_roi_valid']!='1'],'model_inference_executed':False,'visual_review':'pending'}
 dump(batch/'verification.json',report);dump(DATA/'dataset_version.json',report)
 print(f"PASS: processed {len(pending)}; cumulative {len(rows)}; no model inference")

if __name__=='__main__':main()
