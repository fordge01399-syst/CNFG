from pathlib import Path
import json, statistics
import matplotlib.pyplot as plt
ROOT=Path('/home/ubuntu/cnfg_agent'); R=ROOT/'results'; P=ROOT/'plots'; P.mkdir(exist_ok=True)
s=json.loads((R/'corrected_summary_fixed.json').read_text())
models=['cnfg','gru','mlp']; labels=['CNFG-Agent','GRU','MLP']; colors=['#1f77b4','#d62728','#2ca02c']
# main success plot
splits=['iid_test','novel_layout_test','novel_task_test','long_horizon_test','partial_observation_test']; titles=['IID','Novel layout','Novel task','Long horizon','Partial observation']
fig,ax=plt.subplots(figsize=(10,5));
for m,l,c in zip(models,labels,colors):
 vals=[s[m][sp]['full_task_success']['mean'] for sp in splits]; err=[s[m][sp]['full_task_success']['sd'] for sp in splits]; ax.errorbar(titles,vals,yerr=err,marker='o',label=l,color=c,capsize=4,linewidth=2)
ax.set_ylim(0,1); ax.set_ylabel('Full Task Success'); ax.set_title('Closed-loop task success, five seeds'); ax.legend(); ax.tick_params(axis='x',rotation=20); fig.tight_layout(); fig.savefig(P/'closed_loop_success.png',dpi=180); plt.close(fig)
# horizon
fig,ax=plt.subplots(figsize=(7,4.5)); vals=[s[m]['long_horizon_test']['full_task_success']['mean'] for m in models]; err=[s[m]['long_horizon_test']['full_task_success']['sd'] for m in models]; ax.bar(labels,vals,yerr=err,capsize=4,color=colors); ax.set_ylim(0,1); ax.set_ylabel('Full Task Success'); ax.set_title('Long-horizon closed-loop performance'); fig.tight_layout(); fig.savefig(P/'long_horizon_success.png',dpi=180); plt.close(fig)
# memory
rows=[]
for gap in [1,4,8]:
 rs=[json.loads((R/f'memory_gap{gap}_seed{seed}.json').read_text()) for seed in range(42,47)]; rows.append((gap,statistics.mean([r['full_task_success'] for r in rs]),statistics.pstdev([r['full_task_success'] for r in rs])))
fig,ax=plt.subplots(figsize=(6,4)); ax.errorbar([r[0] for r in rows],[r[1] for r in rows],yerr=[r[2] for r in rows],marker='o',capsize=4); ax.set_xlabel('Full-observation steps before hiding key/door'); ax.set_ylabel('Full Task Success'); ax.set_ylim(0,1); ax.set_title('Memory-gap test'); fig.tight_layout(); fig.savefig(P/'memory_gap_success.png',dpi=180); plt.close(fig)
