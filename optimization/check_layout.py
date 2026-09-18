"""Check flatten identity assets and every convolution gather address."""
from pathlib import Path
import json
import hashlib

root = Path(__file__).resolve().parent.parent
checks = {'flatten_references': [], 'banked_gathers': []}
for folder in (root/'reference').iterdir():
    if not folder.is_dir(): continue
    for pool in folder.glob('**/pool2.txt'):
        flat = pool.with_name('flatten.txt')
        if not flat.exists(): continue
        pv, fv = pool.read_text().split(), flat.read_text().split()
        assert len(pv) == 256 and pv == fv, (pool, flat)
        checks['flatten_references'].append({'pool2':pool.relative_to(root).as_posix(),
                                            'flatten':flat.relative_to(root).as_posix(),
                                            'values':256, 'mismatches':0})
for cin, h, w in ((1,28,28),(6,12,12)):
    count = 0
    for oh in range(h-4):
        for ox in range(0,w-4,8):
            for ic in range(cin):
                for kh in range(5):
                    for kw in range(5):
                        start = (ic*h+oh+kh)*w+ox+kw
                        # Each physical bank contributes one value.
                        bank_addresses = [(start//8 + (b < start%8))*8+b for b in range(8)]
                        expected = [((ic*h+oh+kh)*w+ox+m+kw) for m in range(8)]
                        actual = [bank_addresses[(start%8+m)%8] for m in range(8)]
                        assert actual == expected
                        assert all(0 <= p < cin*h*w for p in bank_addresses)
                        count += 8
    checks['banked_gathers'].append({'input_shape':[cin,h,w], 'checked_addresses':count,
                                      'mismatches':0,'out_of_bounds':0})
assert checks['flatten_references'], 'No source flatten references found'
for c in range(16):
    for h in range(4):
        for w in range(4):
            assert c*16+h*4+w == (c*4+h)*4+w
checks['pool2_fc1_shape'] = [16,4,4]
checks['fc1_input_index'] = '16*c + 4*h + w'
(root/'optimization/results/layout_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
