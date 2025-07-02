# pages/5_Regression.py
import streamlit as st, pandas as pd, matplotlib.pyplot as plt
from utils import load_data, encode_categoricals
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
st.header("Regression")
df=load_data()
target=st.selectbox("Target", {'Q16':'Monthly Willingness to Pay','Q18':'Last 12m Spend'})
feature_cols=[c for c in df.columns if c not in [target,'Q9']]
df=df.dropna(subset=[target])
X, enc=encode_categoricals(df[feature_cols],exclude=['Q11','Q14','Q20','Q21'])
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
models={'Linear':LinearRegression(),'Ridge':Ridge(),'Lasso':Lasso(alpha=0.1),'DT':DecisionTreeRegressor(max_depth=6)}
metrics=[]
for n,m in models.items():
    m.fit(X_train,y_train); pred=m.predict(X_test)
    metrics.append({'Model':n,'MAE':mean_absolute_error(y_test,pred),'R2':r2_score(y_test,pred)})
mdf=pd.DataFrame(metrics).set_index('Model')
st.dataframe(mdf)
best=mdf['R2'].idxmax()
model=models[best]
pred=model.predict(X_test)
fig, ax=plt.subplots(); ax.scatter(y_test,pred); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(best)
st.pyplot(fig)
