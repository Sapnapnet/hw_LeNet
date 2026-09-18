"""Replay the passing HLS RTL testbench with XSim SAIF logging enabled."""
from pathlib import Path
import argparse
import subprocess
import hashlib
import json
import re

p = argparse.ArgumentParser()
p.add_argument('variant', choices=('baseline', 'optimized', 'aggressive5b'))
a = p.parse_args()
root = Path(__file__).resolve().parent.parent
solution = root/'optimization/build'/a.variant/'solution1'
report = solution/'sim/report/lenet_accelerator_with_weights_cosim.rpt'
if not report.exists() or 'Pass' not in report.read_text():
    raise SystemExit('A passing C/RTL co-simulation report is required before replay.')
sim = solution/'sim/verilog'
out = root/'optimization/results'/a.variant
# HLS may speculatively read one past a logical memory's depth after the last
# useful transfer. Its checking model produces X for those reads. For power,
# use a memory wrapper that gates invalid reads and holds its previous output.
# Keep the original checker files and all DUT RTL unchanged.
models=sim/'power_memory_models'
models.mkdir(exist_ok=True)
prj=(sim/'lenet_accelerator_with_weights.prj').read_text()
model_count=0
for memory in sim.glob('AESL_automem_*.v'):
    if memory.name == 'AESL_automem_logits_V.v':
        continue  # write-only output checker, never drives a DUT data input
    content=memory.read_text()
    for port in (0,1):
        old=f'dout{port} <= #DLY mem[address{port}];'
        new=f'if (address{port} < DEPTH) dout{port} <= #DLY mem[address{port}];'
        assert old in content, f'Unexpected memory model: {memory}'
        content=content.replace(old,new)
    (models/memory.name).write_text(content,encoding='ascii')
    prj=prj.replace('"'+memory.name+'"','"power_memory_models/'+memory.name+'"')
    model_count+=1
assert model_count>0 and 'power_memory_models/' in prj
(sim/'power_activity.prj').write_text(prj,encoding='ascii')
recording = out/'activity_recording.saif'
saif = recording.as_posix()
tb = 'apatb_lenet_accelerator_with_weights_top/AESL_inst_lenet_accelerator_with_weights'
script = f'''open_saif {{{saif}}}
set observed [get_objects -r /{tb}/*]
puts "SAIF_OBSERVED_OBJECTS [llength $observed]"
if {{[llength $observed] == 0}} {{error "No DUT objects matched"}}
log_saif $observed
run all
close_saif
quit
'''
(sim/'capture_activity.tcl').write_text(script, encoding='ascii')
batch = (sim/'run_xsim.bat').read_text()
batch = batch.replace(' -prj ', ' --debug all -prj ')
batch = batch.replace('-prj lenet_accelerator_with_weights.prj', '-prj power_activity.prj')
batch = batch.replace('-tclbatch lenet_accelerator_with_weights.tcl', '-tclbatch capture_activity.tcl')
(sim/'run_saif.bat').write_text(batch, encoding='ascii')
with (out/'activity_capture.log').open('w') as log:
    run = subprocess.run(['cmd.exe', '/c', str(sim/'run_saif.bat')], cwd=sim,
                         stdout=log, stderr=subprocess.STDOUT)
if run.returncode:
    raise SystemExit(run.returncode)
if not recording.exists():
    raise SystemExit('XSim did not generate activity.saif; inspect activity_capture.log')
text = (out/'activity_capture.log').read_text(errors='replace')
if 'ERROR:' in text or '## close_saif' not in text:
    raise SystemExit('SAIF recording did not close successfully')
gold=solution/'sim/tv/cdatafile/c.lenet_accelerator_with_weights.autotvout_logits_V.dat'
actual=solution/'sim/tv/rtldatafile/rtl.lenet_accelerator_with_weights.autotvout_logits_V.dat'
expected_tokens=[int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',gold.read_text())]
actual_tokens=[int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',actual.read_text())]
if len(expected_tokens)!=30 or actual_tokens!=expected_tokens:
    raise SystemExit('Power replay logits do not match the original 3-transaction C reference')
recording.replace(out/'activity.saif')
data=(out/'activity.saif').read_bytes()
(out/'activity_capture.complete.json').write_text(json.dumps({
    'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
    'observed_objects':int(re.search(r'SAIF_OBSERVED_OBJECTS (\d+)',text).group(1)),
    'external_memory_model':'hold previous output on out-of-range speculative reads',
    'external_memory_modules':model_count,
    'replay_raw_logits_matched':len(expected_tokens),
    'passing_cosim_report':report.relative_to(root).as_posix()},indent=2))
print(f'SAIF written: {out/"activity.saif"}, {len(data)} bytes')
