# pages/4_Association_Rule_Mining.py
import streamlit as st, pandas as pd
from utils import load_data
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
st.header("Association Rule Mining")
df=load_data()
col=st.selectbox("Select column", ['Q11','Q14','Q20','Q21'])
minsup=st.slider("Min Support",0.01,0.5,0.05,0.01)
minconf=st.slider("Min Confidence",0.1,1.0,0.3,0.05)
transactions=df[col].dropna().astype(str).apply(lambda x:[i.strip() for i in x.split(',')]).tolist()
te=TransactionEncoder(); te_ary=te.fit(transactions).transform(transactions)
tdf=pd.DataFrame(te_ary,columns=te.columns_)
freq=apriori(tdf,min_support=minsup,use_colnames=True)
rules=association_rules(freq,metric="confidence",min_threshold=minconf)
st.dataframe(rules.sort_values('confidence',ascending=False).head(10))
