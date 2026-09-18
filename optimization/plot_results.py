from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root=Path(__file__).resolve().parent.parent
out=root/'optimization/results'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
names=['baseline','optimized','aggressive5b']
labels=['原始工程','高精度版 optimized','低延迟版 aggressive5b']
values=[]
for v in names:
    p=root/'optimization/build'/v/'solution1/sim/report/lenet_accelerator_with_weights_cosim.rpt'
    text=p.read_text()
    values.append(int(re.search(r'\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)',text).group(1)))
fig,ax=plt.subplots(figsize=(8.2,4.5))
fig.subplots_adjust(left=.19,right=.97,bottom=.21,top=.85)
ax.barh(range(len(values)),values,color=['#64748b','#0d9488','#ea580c'],height=.52)
ax.invert_yaxis();ax.set_xlim(0,max(values)*1.23)
ax.set_yticks(range(len(values)),labels,fontproperties=font,fontsize=12)
ax.set_xlabel('RTL latency (cycles at 100 MHz)',fontsize=11)
ax.set_title(f'LeNet 整网延迟最高降低 {100*(1-values[-1]/values[0]):.2f}%（{values[0]/values[-1]:.2f} 倍加速）',fontproperties=font,fontsize=15,pad=14)
for i,v in enumerate(values):
    ax.text(v+1300,i,f'{v:,} cycles\n{v*0.01:.2f} us',va='center',fontsize=10)
ax.spines[['top','right','left']].set_visible(False)
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.text(.08,.055,'Vivado HLS 2018.3 | xc7z020clg400-1 | W12/A12 | 3 RTL transactions per version, all Pass',fontsize=8,color='#475569')
for ext in ('png','svg'):
    fig.savefig(out/f'latency_comparison.{ext}',dpi=180,bbox_inches='tight')
print(out/'latency_comparison.png')
