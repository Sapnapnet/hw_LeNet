"""Collect measured reports; never fill missing evidence with predicted numbers."""
from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'optimization/results'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = {'part': 'xc7z020clg400-1', 'clock_ns': 10,
               'variants': {}, 'source_sha256': {}}
    refs = read_csv(ROOT/'results/accuracy/python_fixed_10000.csv')
    synth_rows = []
    for variant in ('baseline', 'straight', 'noflatten', 'optimized',
                    'fc24b', 'aggressive12b', 'aggressive5b'):
        d = summary['variants'][variant] = {}
        reports = ROOT / 'optimization/build' / variant / 'solution1'
        for p in (reports/'syn/report').glob('*_csynth.xml'):
            tree = ET.parse(p).getroot()
            row = {'variant': variant, 'module': p.stem.removesuffix('_csynth'),
                   'source': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)}
            for field in ('Best-caseLatency', 'Worst-caseLatency', 'Interval-min',
                          'Interval-max', 'EstimatedClockPeriod', 'DSP48E',
                          'BRAM_18K', 'LUT', 'FF'):
                row[field] = tree.findtext('.//' + field)
            synth_rows.append(row)
            if row['module'] == 'lenet_accelerator_with_weights': d['hls_synthesis'] = row
        cr = reports/'sim/report/lenet_accelerator_with_weights_cosim.rpt'
        if cr.exists():
            text = cr.read_text()
            d['cosim'] = {'report': cr.relative_to(ROOT).as_posix(), 'text': text}
        d['accuracy_runs'] = []
        for p in sorted((OUT/variant).glob('*.csv')):
            if not p.name.startswith(('csim_', 'cosim_')): continue
            rows = read_csv(p)
            # A running simulator may have flushed only part of its final row.
            rows = [r for r in rows if all(r.get(k) not in (None, '') for k in
                    ('index', 'label', 'pred', *(f'raw_logit{i}' for i in range(10))))]
            declared = int(p.stem.rsplit('_', 1)[1])
            mismatches = 0
            pred_mismatches = 0
            for i, row in enumerate(rows):
                assert int(row['index']) == i and row['label'] == refs[i]['label']
                pred_mismatches += row['pred'] != refs[i]['pred']
                for k in range(10):
                    mismatches += int(row[f'raw_logit{k}']) != int(refs[i][f'raw_logit{k}'])
            correct = sum(row['label'] == row['pred'] for row in rows)
            d['accuracy_runs'].append({
                'file': p.relative_to(ROOT).as_posix(), 'sha256': sha(p),
                'samples': len(rows), 'declared_samples': declared,
                'complete': len(rows) == declared,
                'correct': correct, 'accuracy': correct/len(rows) if rows else None,
                'raw_logit_mismatches_vs_python': mismatches,
                'prediction_mismatches_vs_python': pred_mismatches})
        power = OUT/variant/'power/activity_power.rpt'
        if power.exists():
            text = power.read_text()
            d['power'] = {'source': power.relative_to(ROOT).as_posix(), 'sha256': sha(power)}
            for field in ('Total On-Chip Power (W)', 'Dynamic (W)', 'Device Static (W)',
                          'Confidence Level', 'Design Nets Matched'):
                m = re.search(r'\|\s*'+re.escape(field)+r'\s*\|\s*([^|]+)\|', text)
                if m: d['power'][field] = m.group(1).strip()
    for folder in ('config', 'hls', 'optimization'):
        for p in (ROOT/folder).rglob('*'):
            if 'build' in p.parts or 'results' in p.parts: continue
            if p.suffix in ('.cpp', '.h', '.tcl', '.py', '.ps1'):
                summary['source_sha256'][p.relative_to(ROOT).as_posix()] = sha(p)
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if synth_rows:
        with (OUT/'synthesis.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=synth_rows[0].keys())
            w.writeheader(); w.writerows(synth_rows)
    print(json.dumps({v: {'synth':d.get('hls_synthesis',{}).get('Worst-caseLatency'),
                         'accuracy':d['accuracy_runs'], 'power':d.get('power')}
                      for v,d in summary['variants'].items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
