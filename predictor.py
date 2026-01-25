# app.py - 简化版本
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sys
import os

# 检查必要的库
try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly 不可用，将使用简化界面")

# 设置页面
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

st.title("📈 等级Logistic回归预测计算器")
st.markdown("使用OrderedModel（等级logistic回归）进行健康风险评估预测。")


# 加载模型函数
@st.cache_resource
def load_model():
    """加载预训练的OrderedModel"""
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        # 基本验证
        required_keys = ['res', 'feature_names']
        for key in required_keys:
            if key not in model_data:
                st.error(f"模型文件缺少必需的键: {key}")
                return None

        return model_data
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None


# 预测函数
def predict_single(features_list, model_data):
    """单个样本预测"""
    try:
        # 转换为DataFrame
        X_new = pd.DataFrame([features_list], columns=model_data['feature_names'])

        # 预测概率
        res = model_data['res']
        predicted_probs = res.predict(X_new)

        # 处理预测结果
        if predicted_probs.ndim == 2:
            probabilities = predicted_probs[0]
        else:
            probabilities = predicted_probs

        # 预测最可能的类别
        predicted_class = np.argmax(probabilities)

        # 获取类别标签
        if 'y_category_labels' in model_data:
            predicted_label = model_data['y_category_labels'][predicted_class]
        elif 'y_categories' in model_data:
            predicted_label = str(model_data['y_categories'][predicted_class])
        else:
            predicted_label = str(predicted_class)

        return {
            'prediction': predicted_class,
            'prediction_label': predicted_label,
            'probabilities': probabilities,
            'class_labels': model_data.get('y_category_labels',
                                           model_data.get('y_categories',
                                                          list(range(len(probabilities))))),
        }
    except Exception as e:
        st.error(f"预测时出错: {str(e)}")
        return None


def main():
    # 加载模型
    model_data = load_model()
    if model_data is None:
        st.stop()

    # 显示基本信息
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**特征数量**: {len(model_data['feature_names'])}")
    with col2:
        n_categories = len(model_data.get('y_category_labels', model_data.get('y_categories', [0, 1])))
        st.info(f"**等级数量**: {n_categories}")

    # 创建输入表单
    st.subheader("输入特征值")
    features = {}

    # 特征映射（使用你的映射）
    feature_mappings = {
        '性别': {0: '女', 1: '男'},
        '年龄分层': {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'},
        '月收入': {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'},
        '文化程度': {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'},
        '婚姻状况': {0: '已婚', 1: '离异/丧偶'},
        'BMI分层': {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}
    }

    # 创建输入字段
    cols = st.columns(2)

    with cols[0]:
        features['性别'] = st.selectbox("性别", options=[0, 1], format_func=lambda x: feature_mappings['性别'][x])
        features['年龄分层'] = st.selectbox("年龄分层", options=[1, 2, 3],
                                            format_func=lambda x: feature_mappings['年龄分层'][x])
        features['月收入'] = st.selectbox("月收入", options=[1, 2, 3, 4],
                                          format_func=lambda x: feature_mappings['月收入'][x])
        features['文化程度'] = st.selectbox("文化程度", options=[1, 2, 3, 4],
                                            format_func=lambda x: feature_mappings['文化程度'][x])
        features['婚姻状况'] = st.selectbox("婚姻状况", options=[0, 1],
                                            format_func=lambda x: feature_mappings['婚姻状况'][x])

    with cols[1]:
        features['查尔森共病指数评估量表'] = st.number_input("查尔森共病指数评估量表", min_value=0, max_value=20,
                                                             value=0)
        features['运动耐量表'] = st.number_input("运动耐量表", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        features['简版老年抑郁量表'] = st.number_input("简版老年抑郁量表", min_value=0, max_value=15, value=5)
        features['简易营养量表MNA-SF'] = st.number_input("简易营养量表MNA-SF", min_value=0, max_value=14, value=8)
        features['BMI分层'] = st.selectbox("BMI分层", options=[4, 3, 2, 1],
                                           format_func=lambda x: feature_mappings['BMI分层'][x])

    # 预测按钮
    if st.button("开始预测", type="primary"):
        # 确保特征顺序与模型一致
        features_list = []
        for feature_name in model_data['feature_names']:
            if feature_name in features:
                features_list.append(features[feature_name])
            else:
                st.error(f"特征 {feature_name} 未找到")
                return

        # 进行预测
        with st.spinner("正在计算..."):
            result = predict_single(features_list, model_data)

        if result:
            # 显示结果
            st.success("预测完成！")

            # 预测等级卡片
            st.markdown("### 预测结果")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("预测等级", result['prediction_label'])
                st.metric("置信度", f"{result['probabilities'][result['prediction']]:.2%}")

            with col2:
                # 概率表格
                prob_df = pd.DataFrame({
                    '等级': result['class_labels'],
                    '概率': result['probabilities']
                })
                prob_df['概率 (%)'] = (prob_df['概率'] * 100).round(2)

                # 高亮预测的等级
                def highlight_max(row):
                    return ['background-color: yellow' if row.name == result['prediction'] else '' for _ in row]

                st.dataframe(prob_df[['等级', '概率 (%)']].style.apply(highlight_max, axis=1))

            # 如果Plotly可用，显示图表
            if PLOTLY_AVAILABLE:
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure(data=[
                        go.Bar(
                            x=result['class_labels'],
                            y=result['probabilities'],
                            marker_color=['#FF6B6B' if i == result['prediction'] else '#4ECDC4'
                                          for i in range(len(result['class_labels']))],
                        )
                    ])

                    fig.update_layout(
                        title="等级概率分布",
                        xaxis_title="等级",
                        yaxis_title="概率",
                        yaxis=dict(tickformat=".0%"),
                        height=300
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"无法显示图表: {e}")

            # 显示输入的特征值
            with st.expander("查看输入的特征值"):
                input_data = []
                for feature_name in model_data['feature_names']:
                    value = features[feature_name]
                    if feature_name in feature_mappings and value in feature_mappings[feature_name]:
                        display_value = f"{value} ({feature_mappings[feature_name][value]})"
                    else:
                        display_value = str(value)

                    input_data.append({
                        '特征': feature_name,
                        '输入值': display_value
                    })

                st.table(pd.DataFrame(input_data))


if __name__ == "__main__":
    main()