import pandas as pd, numpy as np, glob, sys
root=sys.argv[1]
F=["proto","tag","pkt_len","flow_packet_count","flow_byte_count","flow_duration","flow_packets_per_second","inter_arrival_time","cpu","mem","tx_MB","rx_MB"]
rows=[]
for f in sorted(glob.glob(root+"/dataset_runs/*/live_dataset.csv"))+sorted(glob.glob(root+"/results/*/live_dataset.csv")):
    d=pd.read_csv(f); n=f.split('/')[-2]
    att=d[d.Label==1]
    rows.append(dict(run=n,rows=len(d),attack=int(d.Label.sum()),attack_pct=round(100*d.Label.mean(),1),
        dur_s=round(d.timestamp.max()-d.timestamp.min(),1),tag1=int((d.tag==1).sum()),
        protos=dict(d.proto.value_counts()),att_dports=sorted(att.dport.unique().tolist()),
        cpu_uniq=d.cpu.nunique(),mem_uniq=d.mem.nunique(),tx_uniq=d.tx_MB.nunique(),
        devs=d.device_id.nunique()))
print(pd.DataFrame(rows).to_string())
d=pd.read_csv(root+"/dataset_final/live_dataset_combined.csv")
print("\ncombined",d.shape, d.Label.value_counts().to_dict())
print(d.groupby('source_file').size())
print("\nper-feature unique values overall:",{c:d[c].nunique() for c in F})
print("\ncpu/mem/tx/rx describe\n",d[["cpu","mem","tx_MB","rx_MB"]].describe().T)
# label vs (proto,dport) purity
print("\nLabel by proto/dport:\n",d.groupby(['proto','dport']).Label.agg(['count','mean']).sort_values('count',ascending=False).head(15))
# same packet seen by multiple switches?
d['k']=d.src_ip.astype(str)+d.dst_ip.astype(str)+d.proto.astype(str)+d.dport.astype(str)
a=d[d.Label==1]
print("\nattack rows per device:",a.device_id.value_counts().to_dict())
print("attack pkt_len:",a.pkt_len.value_counts().head().to_dict(), " benign UDP pkt_len:",d[(d.Label==0)&(d.proto==17)].pkt_len.value_counts().head().to_dict())
