import streamlit as st
import os
import importlib.util

st.set_page_config(page_title="MMORPG Market Analysis Dashboard", layout="wide")
st.markdown("""# 🎮 MMORPG Market Analysis Dashboard

Analyze and visualize synthetic MMORPG player survey data.  
Navigate tabs in the sidebar for:  
- Data Visualisation  
- Classification  
- Clustering  
- Association Rule Mining  
- Regression  
""")

st.sidebar.title("Navigation")
tabs = [
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
]
selected_tab = st.sidebar.radio("Go to Tab", tabs)

pages_dir = "pages"
pages_map = {
    "Data Visualisation": "1_Data_Visualisation.py",
    "Classification": "2_Classification.py",
    "Clustering": "3_Clustering.py",
    "Association Rule Mining": "4_Association_Rule_Mining.py",
    "Regression": "5_Regression.py"
}

def load_page(page_filename):
    page_path = os.path.join(pages_dir, page_filename)
    if os.path.exists(page_path):
        spec = importlib.util.spec_from_file_location("page_module", page_path)
        page_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(page_module)
    else:
        st.error(f"Page {page_filename} not found in 'pages/' folder.")

if selected_tab in pages_map:
    load_page(pages_map[selected_tab])
else:
    st.warning("Please select a tab from the sidebar.")

st.markdown("""<hr><small>This dashboard is powered by Streamlit.<br>
Survey data is synthetic for demonstration and research purposes.</small>""", unsafe_allow_html=True)
