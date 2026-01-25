# app.py - 简化版本，确保能运行
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 设置页面
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

st.title("📈 等级Logistic回归预测计算器")
st.markdown("使用OrderedModel进行健康风险评估预测")


# 加载模型
@st.cache_resource
def load_model():
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        st.success("✅ 模型加载成功")
        return model_data
    except Exception as e:
        st.error(f"❌ 加载模型失败: {str(e)}")
        return None


def main():
    model_data = load_model()
    if model_data is None:
        st.stop()

    # 显示基本信息
    st.sidebar.info(f"特征数量: {len(model_data.get('feature_names', []))}")

    # 输入表单
    st.subheader("输入特征值")

    # 特征映射
    feature_mappings = {
        '性别': {0: '女', 1: '男'},
        '年龄分层': {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'},
        '月收入': {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'},
        '文化程度': {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'},
        '婚姻状况': {0: '已婚', 1: '离异/丧偶'},
        'BMI分层': {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}
    }

    # 收集特征输入
    features = {}

    col1, col2 = st.columns(2)

    with col1:
        features['性别'] = st.selectbox("性别", [0, 1], format_func=lambda x: feature_mappings['性别'][x])
        features['年龄分层'] = st.selectbox("年龄分层", [1, 2, 3],
                                            format_func=lambda x: feature_mappings['年龄分层'][x])
        features['月收入'] = st.selectbox("月收入", [1, 2, 3, 4], format_func=lambda x: feature_mappings['月收入'][x])
        features['文化程度'] = st.selectbox("文化程度", [1, 2, 3, 4],
                                            format_func=lambda x: feature_mappings['文化程度'][x])
        features['婚姻状况'] = st.selectbox("婚姻状况", [0, 1], format_func=lambda x: feature_mappings['婚姻状况'][x])

    with col2:
        features['查尔森共病指数评估量表'] = st.number_input("查尔森共病指数评估量表", 0, 20, 0)
        features['运动耐量表'] = st.number_input("运动耐量表", 0.0, 10.0, 5.0, 0.1)
        features['简版老年抑郁量表'] = st.number_input("简版老年抑郁量表", 0, 15, 5)
        features['简易营养量表MNA-SF'] = st.number_input("简易营养量表MNA-SF", 0, 14, 8)
        features['BMI分层'] = st.selectbox("BMI分层", [4, 3, 2, 1],
                                           format_func=lambda x: feature_mappings['BMI分层'][x])

    # 预测按钮
    if st.button("🚀 开始预测", type="primary"):
        try:
            # 获取特征名称
            feature_names = model_data.get('feature_names', list(features.keys()))

            # 按顺序准备特征值
            features_list = []
            for name in feature_names:
                if name in features:
                    features_list.append(features[name])
                else:
                    st.error(f"缺少特征: {name}")
                    return

            # 转换为DataFrame
            X_new = pd.DataFrame([features_list], columns=feature_names)

            # 预测
            res = model_data['res']
            predicted_probs = res.predict(X_new)

            if predicted_probs.ndim == 2:
                probabilities = predicted_probs[0]
            else:
                probabilities = predicted_probs

            # 获取预测结果
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

            # 结果卡片
            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测等级", predicted_label)
                st.metric("置信度", f"{probabilities[predicted_class]:.2%}")

            with col2:
                # 概率表格
                prob_data = []
                for i, prob in enumerate(probabilities):
                    label = f"等级{i}"
                    if 'y_category_labels' in model_data and i < len(model_data['y_category_labels']):
                        label = model_data['y_category_labels'][i]
                    prob_data.append([label, f"{prob:.2%}"])

                prob_df = pd.DataFrame(prob_data, columns=['等级', '概率'])
                st.dataframe(prob_df, use_container_width=True)

            # 简单图表
            st.subheader("概率分布图")
            chart_df = pd.DataFrame({
                '等级': [f"等级{i}" for i in range(len(probabilities))],
                '概率': probabilities
            })
            st.bar_chart(chart_df.set_index('等级'))

        except Exception as e:
            st.error(f"预测时出错: {str(e)}")


if __name__ == "__main__":
    main()