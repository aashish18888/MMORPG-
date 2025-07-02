# pages/2_Classification.py
import streamlit as st, pandas as pd, numpy as np
from utils import load_data, encode_categoricals
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt, seaborn as sns
st.header("Classification")
df=load_data()
target_col='Q15'
feature_cols=[c for c in df.columns if c not in [target_col,'Q9']]
df=df.dropna(subset=[target_col])
X, enc=encode_categoricals(df[feature_cols],exclude=['Q11','Q14','Q20','Q21'])
y=df[target_col]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
models={'KNN':KNeighborsClassifier(), 'DT':DecisionTreeClassifier(max_depth=6), 'RF':RandomForestClassifier(), 'GBRT':GradientBoostingClassifier()}
metrics={}
for n,m in models.items():
    m.fit(X_train,y_train); pred=m.predict(X_test)
    metrics[n]={'Acc':accuracy_score(y_test,pred),'Prec':precision_score(y_test,pred,average='weighted',zero_division=0),
    'Rec':recall_score(y_test,pred,average='weighted',zero_division=0),'F1':f1_score(y_test,pred,average='weighted',zero_division=0),'pred':pred}
st.dataframe(pd.DataFrame(metrics).T[['Acc','Prec','Rec','F1']])
sel=st.selectbox("Model for CM", list(models.keys()))
cm=confusion_matrix(y_test, metrics[sel]['pred'])
fig, ax=plt.subplots(); sns.heatmap(cm,annot=True,cmap="Blues",ax=ax); ax.set_xlabel("Pred"); ax.set_ylabel("True")
st.pyplot(fig)
