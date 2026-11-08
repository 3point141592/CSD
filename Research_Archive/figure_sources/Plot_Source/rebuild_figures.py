"""Rebuild the manuscript figures from saved experiment records.

Historical plot order is mapped to journal labels (1-4, B1-B3).
The selector chart includes the panel labels used in the manuscript.
No training or inference is performed.
"""
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplcache'))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

D=HERE/'data';F=HERE.parent/'Figures';F.mkdir(exist_ok=True)
BLUE='#0072B2';TEAL='#008578';PURPLE='#7B52AB';GRAY='#596775'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'grid.color':'#E3E7EA','grid.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
def load(s):return pd.read_csv(D/s)
JOURNAL_LABELS={1:'1',2:'2',3:'B1',4:'B2',5:'B3',6:'3',7:'4'}
def save(fig,n):
 for ext in ['png','svg']:fig.savefig(F/f'Figure{JOURNAL_LABELS[n]}.{ext}',dpi=450,bbox_inches='tight',pad_inches=.07)
 plt.close(fig)
def percent(ax):ax.set_ylim(-3,103);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y')
def dots(ax,x,v,color,marker='o',sd=True):
 v=np.asarray(v,float); jitter=np.linspace(-.055,.055,len(v))
 ax.scatter(x+jitter,v,s=9,c=color,alpha=.24,edgecolors='none',zorder=2)
 if sd:ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt=marker,color=color,capsize=3,ms=5,lw=1.2,zorder=4)
 else:ax.plot(x,v.mean(),marker,color=color,ms=6,zorder=4)
def legend(fig,wide=False,prepared=False):
 h=[Line2D([0],[0],color=BLUE,marker='o',ls='',label='Direct: prepared + wide' if prepared else 'Direct: head 16'),Line2D([0],[0],color=TEAL,marker='^',ls='',label='Relational')]
 if wide:h.insert(1,Line2D([0],[0],color=PURPLE,marker='s',ls='',label='Direct: head = dimension'))
 fig.legend(handles=h,loc='upper center',ncol=len(h),frameon=False,bbox_to_anchor=(.5,1.015))
# Use a schematic to explain the protocol without inventing measurements.
fig,ax=plt.subplots(figsize=(7.2,4.55));ax.set_xlim(0,10);ax.set_ylim(0,6.7);ax.axis('off')
def box(x,y,w,h,text,color='#F2F5F7',size=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor=color,edgecolor='#8D9CA8',lw=.8))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.35)
def arr(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1,color=GRAY))
box(.4,5.35,9.2,1.02,'Starting state and supplied component descriptions\nCanonical task: 12 records at 9 centers, including 3 useful destinations')
box(.25,3.6,4.4,1.05,'Direct CSD\nPredict a coordinate direction\nand a travel scale',color='#EAF3FA')
box(5.35,3.6,4.4,1.05,'Relational CSD\nWeight and combine relative vectors\nNormalize and predict a travel scale',color='#EAF5F2')
arr((2.4,5.28),(2.4,4.75));arr((7.6,5.28),(7.6,4.75))
box(.4,2.15,9.2,.76,'Eight initial proposals per model; each initial move is at most 6 units')
arr((2.4,3.5),(2.4,3.0));arr((7.6,3.5),(7.6,3.0))
box(.4,.9,9.2,.70,'Apply the same local refiner to every proposal: 9 steps at rate 0.28')
arr((5,2.07),(5,1.7));arr((5,.83),(5,.37))
ax.text(5,.1,'Measure useful-region coverage and quality across all eight candidates',ha='center',va='center',fontsize=10,fontweight='bold')
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98);save(fig,1)
# Pair each seed's values to expose coordinate sensitivity within a trained model.
p=load('prepared_per_seed.csv');fig,axs=plt.subplots(2,2,figsize=(7.2,5.15),sharex=True,sharey=True)
for c,d in enumerate([32,64]):
 for r,(metric,label) in enumerate([('mode_set_recall','Useful-region coverage (%)'),('useful_candidate_fraction','Useful candidates (%)')]):
  ax=axs[r,c]
  for model,color,offset,mark in [('DirectPrepared',BLUE,-.08,'o'),('Relational',TEAL,.08,'^')]:
   f=p[(p.dimension==d)&(p.model==model)].pivot(index='seed',columns='split',values=metric)
   for jitter,(_,row) in zip(np.linspace(-.055,.055,len(f)),f.iterrows()):ax.plot(np.array([0,1])+offset+jitter,row[['iid','exact_rotated']].to_numpy()*100,color=color,alpha=.13,lw=.6)
   for j,s in enumerate(['iid','exact_rotated']):dots(ax,j+offset,f[s]*100,color,mark)
  percent(ax);ax.set_xlim(-.35,1.35);ax.set_xticks([0,1],['IID tasks','Exact rotation'])
  if c==0:ax.set_ylabel(label)
  ax.set_title(f'{chr(65+r*2+c)}  {d} dimensions',loc='left',fontweight='bold')
legend(fig,prepared=True);fig.subplots_adjust(left=.10,right=.98,bottom=.07,top=.90,wspace=.12,hspace=.22);save(fig,2)
# Separate dimensions visually because the 2D study uses a different generator.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.45))
for ax,metric,label in zip(axs,['mean_matched_cosine','mode_set_recall'],['Initial matched cosine','Useful-region coverage (fraction)']):
 for j,d in enumerate([2,4,8,16,32,64]):
  f=load(f'matched_d{d}.csv');f=f[f.split=='iid']
  for c,col,off,mark in [('CurrentAttention',BLUE,-.11,'o'),('EquivariantAttention',TEAL,.11,'^')]:
   x=f[f.condition==c][metric];ax.errorbar(j+off,x.mean(),yerr=x.std(ddof=1),fmt=mark,color=col,capsize=2,ms=5,lw=1.1)
 ax.axvspan(-.5,.5,color='#F1F2F3',zorder=-1);ax.axvline(.5,color='#A5ADB4',ls=':',lw=1)
 ax.set_ylim(-.03,1.07);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks(range(6),['2*','4','8','16','32','64']);ax.set_xlim(-.5,5.45);ax.set_xlabel('Ambient dimension');ax.set_ylabel(label);ax.grid(axis='y')

for ax,letter in zip(axs,['A','B']): ax.set_title(letter,loc='left',fontweight='bold')
legend(fig)
fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.18,wspace=.30);save(fig,3)
# Pair each seed's achieved and ceiling values to distinguish fit from span limits.
f=load('rank_per_seed.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),sharey=True)
for ax,d,letter in zip(axs,[32,64],['A','B']):
 x=f[f.dimension==d]
 for _,r in x.iterrows():ax.plot([0,1],[r.observed_saved_matched_cosine,r.mean_span_cosine_upper_bound],color=PURPLE,alpha=.22,lw=.8)
 ax.plot([0,1],[x.observed_saved_matched_cosine.mean(),x.mean_span_cosine_upper_bound.mean()],color=PURPLE,marker='D',lw=2,ms=5)
 ax.set_title(f'{letter}  {d}D · measured rank 17',loc='left',fontweight='bold');ax.set_xticks([0,1],['Achieved','Output-span ceiling']);ax.set_xlim(-.2,1.2);ax.set_ylim(.58,.97);ax.grid(axis='y')
axs[0].set_ylabel('Initial directional cosine');fig.subplots_adjust(left=.1,right=.98,top=.88,bottom=.14,wspace=.15);save(fig,4)
# Use full-precision source rows to avoid compounding table-rounding error.
f=load('wide_per_seed.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,3.5))
for ax,metric,label in zip(axs,['mean_matched_cosine','mode_set_recall'],['Initial matched cosine','Useful-region coverage (fraction)']):
 for j,d in enumerate([32,64]):
  for m,col,off,mark in [('Direct16_archived',BLUE,-.23,'o'),('DirectWide',PURPLE,0,'s'),('Relational_archived',TEAL,.23,'^')]:
   v=f[(f.dimension==d)&(f.condition==m)&(f.split=='iid')][metric];dots(ax,j+off,v,col,mark)
 ax.set_xticks([0,1],['32','64']);ax.set_xlabel('Ambient dimension');ax.set_ylabel(label);ax.set_ylim(-.04,1.09);ax.set_xlim(-.5,1.5);ax.grid(axis='y')

for ax,letter in zip(axs,['A','B']): ax.set_title(letter,loc='left',fontweight='bold')
legend(fig,True);fig.subplots_adjust(left=.1,right=.98,top=.82,bottom=.17,wspace=.30);save(fig,5)
# Show individual seeds so their variation remains visible alongside each mean.
s=load('selectors_per_seed.csv');m=load('multiplicity_per_set.csv').groupby('seed').mean(numeric_only=True)
methods=['relational_csd','amplitude_top3_repeat','amplitude_distance_top3_repeat','descriptor_logistic_top3_repeat','amplitude_unique8','random_top3_repeat','multiplicity']
labels=['Relational CSD','Amplitude','Amplitude / distance','Learned affine scorer','All 8 non-anchor centers','Random 3 centers','Multiplicity (CPU)']
fig,axs=plt.subplots(1,3,figsize=(8.1,3.8),sharey=True)
for ax,key,title,mkey in zip(axs,['after_mode_set_recall','after_candidate_useful_rate','after_candidate_decoy_rate'],['Coverage (%)','Useful candidates\n(%)','Decoy candidates\n(%)'],['after_mode_set_recall','after_useful_candidate_rate','after_decoy_rate']):
 for j,method in enumerate(methods):
  vals=m[mkey].to_numpy()*100 if method=='multiplicity' else s[s.method==method][key].to_numpy()*100
  color=TEAL if method=='relational_csd' else PURPLE if method=='multiplicity' else GRAY
  ax.scatter(vals,j+np.linspace(-.12,.12,len(vals)),s=8,color=color,alpha=.25,edgecolors='none')
  ax.plot(vals.mean(),j,marker='D' if method=='multiplicity' else 'o',color=color,ms=5,mfc='white' if method=='multiplicity' else color)
 ax.axhline(5.5,color='#9CA6AE',ls=':',lw=1);ax.set_xlim(-5,105);ax.set_xticks([0,50,100]);ax.set_title(title);ax.grid(axis='x');ax.set_yticks(range(7),labels);ax.set_ylim(6.5,-.6)
# Panel identifiers are part of the standalone image, as required by its revised caption.
for ax,letter in zip(axs,['A','B','C']):
 ax.text(.5,1.20,letter,transform=ax.transAxes,ha='center',va='bottom',fontsize=11,fontweight='bold')
fig.subplots_adjust(left=.31,right=.98,bottom=.08,top=.88,wspace=.14);save(fig,6)
# Separate nested cases from seeds to avoid implying independent replicate counts.
c=load('corruption_per_seed.csv');cone=load('cone_per_set.csv');fig,axs=plt.subplots(1,2,figsize=(8.1,3.5),gridspec_kw={'width_ratios':[1,1.15]})
conds=['clean','add_25pct_irrelevant','add_50pct_irrelevant','remove_goal_landmark_pair'];labels=['All references','+3 distractors','+6 distractors','Remove peak +\ncorridor records']
for j,cond in enumerate(conds):
 v=c[c.condition==cond].mode_set_recall.to_numpy()*100
 axs[0].scatter(v,j+np.linspace(-.12,.12,len(v)),s=8,color=TEAL,alpha=.25)
 axs[0].plot(v.mean(),j,'o',color=TEAL,ms=5)
axs[0].set_yticks(range(4),labels);axs[0].set_ylim(3.5,-.5);axs[0].set_xlim(-2,103);axs[0].set_xticks([0,25,50,75,100]);axs[0].grid(axis='x');axs[0].set_xlabel('Useful-region coverage (%)');axs[0].set_title('A  Reference conditions (32D)',loc='left',fontweight='bold')
axs[1].hist(cone.cone_max_cosine,bins=np.linspace(.30,.80,26),color=PURPLE,edgecolor='white',lw=.4)
axs[1].axvline(cone.cone_max_cosine.mean(),color='#262C31',ls='--',lw=1)
axs[1].text(.96,.94,'900 nested cases\n30 seeds\nMean = 0.496',transform=axs[1].transAxes,ha='right',va='top',fontsize=9)
axs[1].set_xlim(.3,1);axs[1].set_xlabel('Best initial cone-direction cosine');axs[1].set_ylabel('Number of cases');axs[1].set_title('B  Missing-target direction',loc='left',fontweight='bold')
fig.subplots_adjust(left=.18,right=.98,bottom=.17,top=.87,wspace=.42);save(fig,7)
print('Rebuilt Figures 1-4 and B1-B3 from unchanged saved records as PNG and SVG.')
