# 极简测试
import streamlit as st

st.set_page_config(page_title="测试", layout="wide")

st.title("✅ 应用启动成功")
st.write("如果看到这个，说明基础环境正常。")

try:
    import pandas as pd
    st.success("pandas 导入成功")
except Exception as e:
    st.error(f"pandas 导入失败: {e}")

try:
    import numpy as np
    st.success("numpy 导入成功")
except Exception as e:
    st.error(f"numpy 导入失败: {e}")

try:
    import statsmodels.api as sm
    st.success("statsmodels 导入成功")
except Exception as e:
    st.error(f"statsmodels 导入失败: {e}")

try:
    with open('ordered_logit_model.pkl', 'rb') as f:
        import pickle
        model = pickle.load(f)
    st.success("模型加载成功")
    st.write("模型键:", list(model.keys()))
except Exception as e:
    st.error(f"模型加载失败: {e}")
