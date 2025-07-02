# pages/3_Clustering.py
import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from utils import load_data, scale_features
from sklearn.cluster import KMeans
st.header("Clustering")
df=load_data()
feat=['Q1','Q3','Q6','Q7','Q12','Q13','Q15','Q16','Q18','Q22','Q23']
X_scaled,_=scale_features(df[feat])
wcss=[]
for k in range(2,11):
    wcss.append(KMeans(n_clusters=k,n_init=10,random_state=42).fit(X_scaled).inertia_)
fig, ax=plt.subplots(); ax.plot(range(2,11),wcss,marker='o'); ax.set_xlabel("k"); ax.set_ylabel("WCSS")
st.pyplot(fig)
k=st.slider("Clusters",2,10,3)
km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X_scaled)
df['Cluster']=km.labels_
st.dataframe(df[['Cluster']+feat].groupby('Cluster').mean())
st.download_button("Download Labeled", df.to_csv(index=False), "clustered.csv")
