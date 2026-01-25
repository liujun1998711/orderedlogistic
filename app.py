import streamlit as st
import sys
import subprocess
import os


# 检查并安装缺失的包
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


# 检查必要的包
required_packages = [
    'pandas==2.0.3',
    'numpy==1.24.3',
    'statsmodels==0.13.5'
]

missing_packages = []
for package in required_packages:
    pkg_name = package.split('==')[0]
    try:
        __import__(pkg_name)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    with st.spinner(f'正在安装缺失的包: {missing_packages}'):
        for package in missing_packages:
            if install_package(package):
                st.success(f'成功安装 {package}')
            else:
                st.error(f'安装 {package} 失败')
    st.rerun()

# 现在导入必要的库
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 设置页面
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

st.title("📈 等级Logistic回归预测计算器")
st.markdown("使用OrderedModel进行健康风险评估预测")


# 加载模型函数
@st.cache_resource
def load_model():
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("模型文件未找到！请确保 'ordered_logit_model.pkl' 存在。")
        return None
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None


# 主程序
def main():
    model_data = load_model()
    if model_data is None:
        st.stop()

    # 显示模型信息
    st.sidebar.info(f"特征数量: {len(model_data.get('feature_names', []))}")

    # 输入表单
    st.subheader("输入特征值")
    features = {}

    col1, col2 = st.columns(2)

    with col1:
        features['性别'] = st.selectbox("性别", [0, 1], format_func=lambda x: '女' if x == 0 else '男')
        features['年龄分层'] = st.selectbox("年龄分层", [1, 2, 3],
                                            format_func=lambda x: {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'}[x])
        features['月收入'] = st.selectbox("月收入", [1, 2, 3, 4], format_func=lambda x:
        {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'}[x])
        features['文化程度'] = st.selectbox("文化程度", [1, 2, 3, 4], format_func=lambda x:
        {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'}[x])
        features['婚姻状况'] = st.selectbox("婚姻状况", [0, 1], format_func=lambda x: {0: '已婚', 1: '离异/丧偶'}[x])

    with col2:
        features['查尔森共病指数评估量表'] = st.number_input("查尔森共病指数评估量表", 0, 20, 0)
        features['运动耐量表'] = st.number_input("运动耐量表", 0.0, 10.0, 5.0, 0.1)
        features['简版老年抑郁量表'] = st.number_input("简版老年抑郁量表", 0, 15, 5)
        features['简易营养量表MNA-SF'] = st.number_input("简易营养量表MNA-SF", 0, 14, 8)
        features['BMI分层'] = st.selectbox("BMI分层", [4, 3, 2, 1], format_func=lambda x:
        {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}[x])

    if st.button("🚀 开始预测", type="primary"):
        # 获取特征顺序
        feature_names = model_data.get('feature_names', list(features.keys()))
        features_list = [features[name] for name in feature_names]

        # 转换为DataFrame
        X_new = pd.DataFrame([features_list], columns=feature_names)

        # 预测
        res = model_data['res']
        predicted_probs = res.predict(X_new)

        if predicted_probs.ndim == 2:
            probabilities = predicted_probs[0]
        else:
            probabilities = predicted_probs

        predicted_class = np.argmax(probabilities)

        # 获取类别标签
        if 'y_category_labels' in model_data:
            predicted_label = model_data['y_category_labels'][predicted_class]
        elif 'y_categories' in model_data:
            predicted_label = str(model_data['y_categories'][predicted_class])
        else:
            predicted_label = str(predicted_class)

        # 显示结果
        st.success("✅ 预测完成！")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测等级", predicted_label)
            st.metric("置信度", f"{probabilities[predicted_class]:.2%}")

        with col2:
            prob_data = []
            for i, prob in enumerate(probabilities):
                label = f"等级{i}"
                if 'y_category_labels' in model_data and i < len(model_data['y_category_labels']):
                    label = model_data['y_category_labels'][i]
                prob_data.append([label, f"{prob:.2%}"])

            st.table(pd.DataFrame(prob_data, columns=['等级', '概率']))


if __name__ == "__main__":
    main()