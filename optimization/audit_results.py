"""Independent final audit of reports, actual CSV values and SAIF provenance."""
from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parent.parent
evidence={}
for variant in ('baseline','optimized'):
    solution=root/'optimization/build'/variant/'solution1'
    tree=ET.parse(solution/'syn/report/lenet_accelerator_with_weights_csynth.xml').getroot()
    cosim=(solution/'sim/report/lenet_accelerator_with_weights_cosim.rpt').read_text()
    match=re.search(r'\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',cosim)
    assert match and len(set(match.groups()))==1
    result=root/'optimization/results'/variant
    marker=json.loads((result/'activity_capture.complete.json').read_text())
    assert marker['replay_raw_logits_matched']==30
    saif=result/'activity.saif'
    assert hashlib.sha256(saif.read_bytes()).hexdigest()==marker['sha256']
    power=result/'power/activity_power.rpt'
    assert power.stat().st_mtime>saif.stat().st_mtime
    text=power.read_text()
    assert 'Design State     : routed' in text and 'activity.saif' in text
    watts=float(re.search(r'\| Total On-Chip Power \(W\)\s*\|\s*(\d+\.\d+)',text).group(1))
    evidence[variant]={'rtl_cycles':int(match.group(1)),
                       'hls_max_cycles':int(tree.findtext('.//Worst-caseLatency')),
                       'power_estimate_w':watts,'saif_sha256':marker['sha256']}
assert evidence['optimized']['rtl_cycles']<evidence['baseline']['rtl_cycles']
actual=root/'optimization/results/optimized/csim_10000.csv'
reference=root/'results/accuracy/python_fixed_10000.csv'
assert actual.read_bytes()==reference.read_bytes()
with actual.open() as f:
    rows=list(csv.DictReader(f))
assert len(rows)==10000 and sum(row['label']==row['pred'] for row in rows)==9894
evidence['accuracy']={'samples':10000,'correct':9894,'raw_outputs_identical':100000}
# Low-latency variant aggressive5b: channel-pruned, verified against the
# offline ablation model instead of the unpruned python reference.
ablation=json.loads((root/'optimization/results/channel_ablation.json').read_text())
solution5=root/'optimization/build/aggressive5b/solution1'
tree5=ET.parse(solution5/'syn/report/lenet_accelerator_with_weights_csynth.xml').getroot()
cosim5=(solution5/'sim/report/lenet_accelerator_with_weights_cosim.rpt').read_text()
match5=re.search(r'\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)',cosim5)
assert match5 and len(set(match5.groups()))==1
with (root/'optimization/results/aggressive5b/csim_10000.csv').open() as f:
    rows5=list(csv.DictReader(f))
correct5=sum(row['label']==row['pred'] for row in rows5)
assert len(rows5)==10000 and correct5==round(ablation['final_aggressive5b_5x12']*10000)
evidence['aggressive5b']={'rtl_cycles':int(match5.group(1)),
                          'hls_max_cycles':int(tree5.findtext('.//Worst-caseLatency')),
                          'accuracy_samples':10000,'accuracy_correct':correct5,
                          'conv1_keep':ablation['conv1_keep'],'conv2_keep':ablation['conv2_keep']}
power5=root/'optimization/results/aggressive5b/power/activity_power.rpt'
if power5.exists():
    marker5=json.loads((root/'optimization/results/aggressive5b/activity_capture.complete.json').read_text())
    assert marker5['replay_raw_logits_matched']==30
    saif5=root/'optimization/results/aggressive5b/activity.saif'
    assert hashlib.sha256(saif5.read_bytes()).hexdigest()==marker5['sha256']
    assert power5.stat().st_mtime>saif5.stat().st_mtime
    text5=power5.read_text()
    assert 'Design State     : routed' in text5 and 'activity.saif' in text5
    evidence['aggressive5b']['power_estimate_w']=float(re.search(r'\| Total On-Chip Power \(W\)\s*\|\s*(\d+\.\d+)',text5).group(1))
    evidence['aggressive5b']['saif_sha256']=marker5['sha256']
baseline=(root/'optimization/build/baseline/solution1/syn/report/lenet_accelerator_with_weights_csynth.rpt').read_text()
optimized=(root/'optimization/build/optimized/solution1/syn/report/lenet_accelerator_with_weights_csynth.rpt').read_text()
assert 'flatten_buffer' in baseline and 'flatten_buffer' not in optimized
straight=ET.parse(root/'optimization/build/straight/solution1/syn/report/lenet_accelerator_with_weights_csynth.xml').getroot()
noflatten=ET.parse(root/'optimization/build/noflatten/solution1/syn/report/lenet_accelerator_with_weights_csynth.xml').getroot()
assert int(straight.findtext('.//Worst-caseLatency'))-int(noflatten.findtext('.//Worst-caseLatency'))==258
evidence['flatten']={'removed_buffer_in_synthesized_top':True,'isolated_saving_cycles':258}
for doc in (root/'optimization/REPORT_CN.md',root/'optimization/README.md',root.parent/'README_优化.md'):
    for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if link.startswith('http'):continue
        assert (doc.parent/link).exists(),(doc,link)
(root/'optimization/results/completion_audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps(evidence,indent=2))
