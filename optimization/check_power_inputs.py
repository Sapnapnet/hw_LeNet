"""Check known input values over the complete power simulation window."""
from pathlib import Path
import re
import json

root=Path(__file__).resolve().parent.parent
results={}
for variant in ('baseline','optimized'):
    values=[]
    names=set()
    saif=root/'optimization/results'/variant/'activity.saif'
    for line in saif.read_text().splitlines():
        match=re.match(r'\s*\(((?:input_V|conv[12]_w_\d+_V|fc[123]_w_\d+_V)_q\d\\\[\d+\\\])',line)
        if match:
            names.add(match.group(1))
            values.append(int(re.search(r'\(TX (\d+)\)',line).group(1)))
    assert len(names)==576 and not any(values), (variant,len(names),sum(values))
    results[variant]={'unique_input_data_bit_names':len(names),
                      'hierarchical_observations':len(values),
                      'observations_with_unknown_time':0}
(root/'optimization/results/power_input_quality.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
