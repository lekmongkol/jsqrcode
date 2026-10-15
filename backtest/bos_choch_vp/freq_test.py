"""Trade frequency check: best rules (H4 trend + VP-leg P/D + 2R + BE at OB) on M15 / M30 / H1."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
import backtest as b, filters as fl, pandas as pd, numpy as np
from exit_test import stats
d12,d26=b.load_2012(),b.load_2026()
yrs={'2012-2017':5.6,'2018-2022':4.2,'2026':0.5}
allf={}
for tf,rule,mins in (('M15',None,15),('M30','30min',30),('H1','1h',60)):
  for lvl in (0.5,0.382):
    fs=[]
    for name,base,clock in (('old',d12,'server'),('new',d26,'utc')):
      df=base if rule is None else b.resample(base,rule)
      df=fl.add_htf_cols(fl.add_session_cols(df,clock,mins),base,mins)
      sp=0.3 if name=='old' else 0.5
      f=b.to_frame(b.run(df.assign(htf=df.htf_struct),b.Params(pivot=3,rr=2.0,ob='be_at_ob',trend='with',pd_mode='leg',pd_level=lvl,spread=sp,min_risk_atr=0.5)))
      fs.append(f)
    f=pd.concat(fs); allf[(tf,lvl)]=f
    s=stats(f); per=f.groupby(f.entry_time.dt.year<2018).r.agg(lambda r: round(r[r>0].sum()/-r[r<0].sum(),2))
    print(tf,lvl,'trades',s['trades'],'per_yr',round(s['trades']/10.3,1),'PF',s['PF'],'avgR',s['avgR'],'t',s['t'],'PF<2018',per.get(True),'PF>=2018',per.get(False))
f=pd.concat([allf[(t,0.5)] for t in ('M15','M30','H1')]); s=stats(f)
print('COMBINED M15+M30+H1 @0.5','trades',s['trades'],'per_yr',round(s['trades']/10.3,1),'PF',s['PF'],'avgR',s['avgR'],'t',s['t'],'maxDD',s['maxDD'])
