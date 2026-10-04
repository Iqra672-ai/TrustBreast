import os
import matplotlib
if 'ipykernel' not in __import__('sys').modules: matplotlib.use('Agg')
def _root():
    c=[]
    if '__file__' in globals(): c.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    c+= [os.getcwd(), os.path.join(os.getcwd(),'TrustBreast'), '/content/TrustBreast', os.path.dirname(os.getcwd())]
    for p in c:
        if os.path.isdir(os.path.join(p,'results')): return p
    return os.getcwd()
ROOT=_root()
os.makedirs(os.path.join(ROOT,'figures'),exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Serif'})
fig=plt.figure(figsize=(19.6,20.2)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1960); ax.set_ylim(2020,0); ax.axis('off')
def box(x,y,w,h,fc='white',lw=1.5): ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec='#333',lw=lw))
def hdr(x,y,w,t,h=44): box(x,y,w,h,'#e8e8e8'); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=16,fontweight='bold')
def arrow(x,y1,y2): ax.annotate('',xy=(x,y2),xytext=(x,y1),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1.6,mutation_scale=18))
T=lambda x,y,s,**k: ax.text(x,y,s,ha=k.pop('ha','center'),va=k.pop('va','center'),**k)
T(980,50,'TrustBreast — Complete Framework Roadmap',fontsize=24,fontweight='bold')
T(980,95,'WBCD · 569 patients · 30 FNA features · binary classification',fontsize=14,style='italic')
ax.plot([370,1590],[130,130],color='#333',lw=1.2)
box(425,175,1110,92); T(980,205,'Wisconsin Breast Cancer Diagnostic Dataset (WBCD)',fontsize=16,fontweight='bold'); T(980,240,'569 patients · 30 FNA features · 212 malignant (37.3%) · 357 benign (62.7%)',fontsize=13)
arrow(980,267,310)
box(145,312,1670,196,'#fafafa'); hdr(145,312,1670,'Preprocessing Pipeline')
for i,(a,b) in enumerate([('MinMaxScaler','[0, 1]; fitted on training data only'),('SMOTE','applied to training data only (inside each fold)'),('Gaussian noise augmentation','×3 (σ = 0.01 and σ = 0.02)')]):
    x=208+i*512; box(x,380,512,104); T(x+256,412,a,fontsize=15,fontweight='bold'); T(x+256,448,b,fontsize=12.5)
arrow(980,508,550)
box(145,552,1670,288,'#fafafa'); hdr(145,552,1670,'O1 — Leakage-Free Soft-Voting Ensemble')
for i,(a,b,c) in enumerate([('Random Forest','n_estimators = 500','max_depth = None'),('XGBoost','n_estimators = 500','learning_rate = 0.01, depth = 4'),('Deep Neural Network','1024 → 512 → 256 → 128','→ 64 → 32 → 1 (sigmoid)')]):
    x=208+i*512; box(x,620,512,124); T(x+256,652,a,fontsize=15,fontweight='bold'); T(x+256,686,b,fontsize=12.5); T(x+256,712,c,fontsize=12.5)
T(980,775,'P(ensemble) = [ P(RF) + P(XGB) + P(DNN) ] / 3        thresholds tuned on validation only: RF 0.43 · XGB 0.39 · DNN 0.36 · ensemble 0.35',fontsize=13,style='italic')
T(980,808,'held-out split 99.12% · 30 repeated splits 97.19 ± 1.65% · 10-fold CV 96.49 ± 2.74% · logistic regression 97.48 ± 2.20%',fontsize=12.5,style='italic',color='#444')
for x in (400,980,1560): arrow(x,840,882)
cols=[('O2 SHAP + LIME Explainability',['– TreeExplainer → RF and XGBoost (exact)','– DeepExplainer → DNN','– SHAP ÷ global std before averaging','– LIME: 15,000 perturbations × 3 runs','– Spearman ρ = 0.95 (top-10); 8/10 overlap','– Drop-column: texture family largest (n.s.) loss'],
       'Worst texture ranks 4th in ensemble SHAP\nand 4th in LIME, while every texture\nfeature is outside the linear top-10.'),
      ('O3 Uncertainty Quantification',['– MC Dropout: T = 100 stochastic passes','– 12 of 114 patients exceed mc_std = 0.15','– ECE = 0.0474 (descriptive)','– Split-conformal (91 cal.): 93.86% marginal','– Cross-conformal (569 OOF, held-out folds):','   95.08% marginal, 94.81% malignant'],
       'Escalation set: the 12 flagged patients\n(10.5%) contain every error made by the\nensemble and by its neural component.'),
      ('O4 DiCE Counterfactuals',['– K = 3 CFs × 12 high-uncertainty patients','– Seed-locked; aligned to the 0.35 threshold','– 17/36 pass both plausibility checks','– Mean L1: 0.7543 all / 0.6062 plausible','– Most changed: texture_worst (27.8%),','   perimeter_worst (25.0%)'],
       'Coherence finding: the four most-changed\nDiCE features are the four highest-ranked\nensemble SHAP features.')]
for i,(h,items,foot) in enumerate(cols):
    x=145+i*580; box(x,884,512,500,'#fafafa'); hdr(x,884,512,h)
    for j,t in enumerate(items): T(x+25,965+j*36,t,fontsize=12.5,ha='left')
    ax.plot([x+30,x+482],[1195,1195],color='#aaa',lw=1)
    T(x+256,1265,foot,fontsize=12.5,style='italic',linespacing=1.5)
for x in (400,980,1560): arrow(x,1384,1426)
box(283,1428,1394,136,'#fafafa'); hdr(283,1428,1394,'Clinically Complete Per-Patient Output')
T(980,1502,'malignancy probability · MC-Dropout interval and escalation flag · conformal prediction set',fontsize=13.5); T(980,1535,'SHAP / LIME explanation · DiCE counterfactual (measurement sensitivity)',fontsize=13.5)
arrow(980,1564,1606)
box(145,1608,1670,170,'#fafafa'); hdr(145,1608,1670,'Headline Results')
vals=[('97.19 ± 1.65%','30 repeated splits'),('96.49 ± 2.74%','10-fold CV'),('99.12%','held-out split'),('0.9997','AUC (held-out)'),('12 / 114','escalation set'),('95.08%','cross-conformal cov.'),('0.0474','ECE (MC Dropout)')]
w=1670/7
for i,(v,l) in enumerate(vals):
    x=145+i*w
    if i: ax.plot([x,x],[1660,1770],color='#ccc',lw=1)
    T(x+w/2,1698,v,fontsize=16.5,fontweight='bold'); T(x+w/2,1740,l,fontsize=12)
T(145,1815,'Note on leakage correction.',fontsize=13,fontweight='bold',ha='left')
T(145,1880,'SMOTE and scaling are fitted inside each training fold only. Measured on five clinical cohorts, applying SMOTE before\n'
 'partitioning raised cross-validated accuracy on every cohort (+0.24 to +3.67 points) but significantly on none; on WBCD\n'
 'the inflation was +0.46 points. The 99.12% held-out figure is one favourable split (73rd percentile of 30).',fontsize=12,ha='left',color='#333',linespacing=1.5)
out=os.path.join(ROOT,'figures','fig01_framework.png'); fig.savefig(out,dpi=100); print('Saved',out)
