# pages/1_Data_Visualisation.py
import streamlit as st, pandas as pd, plotly.express as px
from utils import load_data
st.header("Data Visualisation")
df=load_data()
age_min, age_max = st.slider("Age Range", int(df['Q1'].min()), int(df['Q1'].max()), (18,35))
income_min, income_max = st.slider("Income Range", int(df['Q3'].min()), int(df['Q3'].max()), (df['Q3'].min(), df['Q3'].max()))
genre_filter = st.multiselect("Genre", df['Q10'].unique(), default=list(df['Q10'].unique()))
f = df[(df['Q1']>=age_min)&(df['Q1']<=age_max)&(df['Q3']>=income_min)&(df['Q3']<=income_max)&(df['Q10'].isin(genre_filter))]
st.dataframe(f.head())
st.download_button("Download Filtered", f.to_csv(index=False), "filtered.csv")
fig1=px.histogram(f,x='Q1'); st.plotly_chart(fig1,use_container_width=True)
