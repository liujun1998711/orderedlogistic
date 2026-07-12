import streamlit as st
import pickle
import pandas as pd
import numpy as np
import traceback

st.set_page_config(page_title="Test", layout="wide")

try:
    st.write("Loading model...")
    with open('ordered_logit_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    st.write("Model loaded")
    st.write("Keys:", list(model_data.keys()))
except Exception as e:
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# 简单测试
try:
    import statsmodels.api as sm
    st.write("statsmodels OK")
except Exception as e:
    st.error(f"statsmodels error: {e}")
    st.stop()

st.write("App ready")
