import os
import matplotlib
if 'ipykernel' not in __import__('sys').modules: matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
def _root():
    c=[]
    if '__file__' in globals(): c.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    c+= [os.getcwd(), os.path.join(os.getcwd(),'TrustBreast'), '/content/TrustBreast', os.path.dirname(os.getcwd())]
    for p in c:
        if os.path.isdir(os.path.join(p,'results')): return p
    return os.getcwd()
ROOT=_root()
os.makedirs(os.path.join(ROOT,'figures'),exist_ok=True)
d=pd.read_csv(os.path.join(ROOT,'results','O3_all_patients.csv') if os.path.exists(os.path.join(ROOT,'results','O3_all_patients.csv')) else 'https://raw.githubusercontent.com/Iqra672-ai/TrustBreast/main/results/O3_all_patients.csv')
mean=d.mc_mean.values; std=d.mc_std.values; lo=d.ci_lower.values; hi=d.ci_upper.values
mal=(d.true_label.values=='M')
RED,BLUE='#e74c3c','#3498db'
fig,ax=plt.subplots(1,3,figsize=(16,5),dpi=150)
ax[0].hist(std,bins=25,color='steelblue',edgecolor='white')
ax[0].axvline(0.05,ls='--',color='orange',lw=1.5,label='Medium (0.05)')
ax[0].axvline(0.15,ls='--',color='red',lw=1.5,label='Escalation (0.15)')
ax[0].set_xlabel('MC Dropout std'); ax[0].set_ylabel('Patients')
ax[0].set_title('Uncertainty distribution (T = 100)',fontweight='bold'); ax[0].legend(); ax[0].grid(alpha=.3)
o=np.argsort(mean); x=np.arange(len(o))
ax[1].fill_between(x,lo[o],hi[o],color='gray',alpha=.25)
ax[1].scatter(x,mean[o],c=[RED if m else BLUE for m in mal[o]],s=18,zorder=3)
ax[1].axhline(0.5,ls='--',color='k',lw=1)
ax[1].legend(handles=[mpatches.Patch(color=RED,label='Malignant'),mpatches.Patch(color=BLUE,label='Benign'),mpatches.Patch(color='gray',alpha=.3,label='95% interval')],loc='upper left')
ax[1].set_xlabel('Patients (sorted by MC mean)'); ax[1].set_ylabel('P(malignant)')
ax[1].set_title('MC mean with 95% interval',fontweight='bold'); ax[1].grid(alpha=.3)
ax[2].scatter(mean,std,c=[RED if m else BLUE for m in mal],s=22)
ax[2].axhline(0.15,ls='--',color='red',lw=1.5,label='Escalation (0.15)')
ax[2].axhline(0.05,ls='--',color='orange',lw=1.5,label='Medium (0.05)')
for p in (16,87):
    ax[2].annotate(f'P{p}',(mean[p],std[p]),xytext=(8,4),textcoords='offset points',color='darkred',fontsize=10)
ax[2].set_xlabel('MC mean P(malignant)'); ax[2].set_ylabel('MC std')
ax[2].set_title(f'Uncertainty landscape ({int((std>0.15).sum())} flagged)',fontweight='bold'); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout()
out=os.path.join(ROOT,'figures','fig09_mc.png'); plt.savefig(out); print('Saved',out)
