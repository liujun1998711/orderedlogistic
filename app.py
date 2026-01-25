# app.py - 完整的Streamlit应用
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

# 页面标题
st.title("📈 等级Logistic回归预测计算器")
st.markdown("使用OrderedModel（等级logistic回归）进行健康风险评估预测")


# 加载模型函数
@st.cache_resource
def load_model():
    """加载预训练的OrderedModel"""
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ 模型文件未找到！请确保 'ordered_logit_model.pkl' 存在。")
        return None
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {str(e)}")
        return None


# 预测函数
def predict_single(features_dict, model_data):
    """单个样本预测"""
    try:
        # 获取特征名称（确保顺序与训练时一致）
        feature_names = model_data['feature_names']

        # 将用户输入转换为模型需要的格式
        features_list = []
        for feature in feature_names:
            value = features_dict[feature]
            features_list.append(value)

        # 转换为DataFrame
        X_new = pd.DataFrame([features_list], columns=feature_names)

        # 使用拟合结果进行预测
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
            'feature_values': features_dict
        }
    except Exception as e:
        st.error(f"预测时出错: {str(e)}")
        return None


# 主函数
def main():
    # 加载模型
    model_data = load_model()
    if model_data is None:
        st.stop()

    # 显示模型信息
    st.sidebar.success("✅ 模型加载成功")
    st.sidebar.info(f"特征数量: {len(model_data.get('feature_names', []))}")

    # 创建两个标签页
    tab1, tab2 = st.tabs(["🔍 单个预测", "ℹ️ 关于"])

    with tab1:
        st.subheader("单个样本预测")

        # 根据你的特征映射创建输入框
        st.markdown("### 请输入特征值:")

        # 创建两列布局
        col1, col2 = st.columns(2)

        features_dict = {}

        with col1:
            # 1. 性别
            gender = st.selectbox(
                "性别",
                options=[0, 1],
                format_func=lambda x: "女" if x == 0 else "男",
                help="0: 女, 1: 男"
            )
            features_dict['性别'] = gender

            # 2. 年龄分层
            age_stratum = st.selectbox(
                "年龄分层",
                options=[1, 2, 3],
                format_func=lambda x: ["60-69岁", "70-79岁", "≥80岁"][x - 1],
                help="年龄分层: 1:60-69岁, 2:70-79岁, 3:≥80岁"
            )
            features_dict['年龄分层'] = age_stratum

            # 3. 月收入
            monthly_income = st.selectbox(
                "月收入",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["<2000元", "2000-4000元", "4000-6000元", ">6000元"][x - 1],
                help="月收入分层"
            )
            features_dict['月收入'] = monthly_income

            # 4. 文化程度
            education = st.selectbox(
                "文化程度",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["小学及以下", "初中", "高中/中专", "大专及以上"][x - 1],
                help="文化程度"
            )
            features_dict['文化程度'] = education

            # 5. 婚姻状况
            marital_status = st.selectbox(
                "婚姻状况",
                options=[0, 1],
                format_func=lambda x: ["已婚", "离异/丧偶"][x],
                help="婚姻状况"
            )
            features_dict['婚姻状况'] = marital_status

        with col2:
            # 6. 查尔森共病指数评估量表
            charlson_index = st.slider(
                "查尔森共病指数评估量表",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="查尔森共病指数评分"
            )
            features_dict['查尔森共病指数评估量表'] = charlson_index

            # 7. 运动耐量表
            exercise_tolerance = st.slider(
                "运动耐量表",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="运动耐量评分"
            )
            features_dict['运动耐量表'] = exercise_tolerance

            # 8. 简版老年抑郁量表
            geriatric_depression = st.slider(
                "简版老年抑郁量表",
                min_value=0,
                max_value=15,
                value=5,
                step=1,
                help="老年抑郁量表评分"
            )
            features_dict['简版老年抑郁量表'] = geriatric_depression

            # 9. 简易营养量表MNA-SF
            mna_sf = st.slider(
                "简易营养量表MNA-SF",
                min_value=0,
                max_value=14,
                value=8,
                step=1,
                help="营养状况评分"
            )
            features_dict['简易营养量表MNA-SF'] = mna_sf

            # 10. BMI分层
            bmi_stratum = st.selectbox(
                "BMI分层",
                options=[4, 3, 2, 1],
                format_func=lambda x: ["偏瘦(<18.5)", "正常(18.5-24)", "超重(24-28)", "肥胖(≥28)"][4 - x],
                help="BMI分层"
            )
            features_dict['BMI分层'] = bmi_stratum

        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                # 验证特征数量
                if len(features_dict) != len(model_data.get('feature_names', [])):
                    st.error("特征数量不匹配！")
                    return

                # 进行预测
                result = predict_single(features_dict, model_data)

                if result is None:
                    return

                # 显示预测结果
                st.success("✅ 预测完成！")

                # 创建结果展示区域
                st.subheader("📋 预测结果")

                # 显示预测等级
                col_result1, col_result2 = st.columns([1, 2])

                with col_result1:
                    # 预测等级卡片
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        margin-bottom: 1rem;
                    ">
                        <h3 style="color: white; margin: 0;">预测等级</h3>
                        <h1 style="font-size: 3rem; margin: 0.5rem 0;">{result['prediction_label']}</h1>
                        <p style="margin: 0;">置信度: {result['probabilities'][result['prediction']]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result2:
                    # 概率表格
                    prob_df = pd.DataFrame({
                        '等级': result['class_labels'],
                        '概率': result['probabilities']
                    })
                    prob_df['概率 (%)'] = (prob_df['概率'] * 100).round(2)

                    # 高亮预测的等级
                    def highlight_row(row):
                        if row.name == result['prediction']:
                            return ['background-color: #e6f7ff'] * len(row)
                        return [''] * len(row)

                    st.dataframe(
                        prob_df[['等级', '概率 (%)']].style.apply(highlight_row, axis=1),
                        use_container_width=True
                    )

                # 显示输入的特征值
                with st.expander("📝 查看输入的特征值"):
                    # 创建特征值表格
                    input_data = []
                    for feature_name, value in features_dict.items():
                        # 根据特征名称显示对应的标签
                        if feature_name == '性别':
                            display_value = f"{value} ({'女' if value == 0 else '男'})"
                        elif feature_name == '年龄分层':
                            mapping = {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'}
                            display_value = f"{value} ({mapping[value]})"
                        elif feature_name == '月收入':
                            mapping = {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'}
                            display_value = f"{value} ({mapping[value]})"
                        elif feature_name == '文化程度':
                            mapping = {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'}
                            display_value = f"{value} ({mapping[value]})"
                        elif feature_name == '婚姻状况':
                            mapping = {0: '已婚', 1: '离异/丧偶'}
                            display_value = f"{value} ({mapping[value]})"
                        elif feature_name == 'BMI分层':
                            mapping = {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}
                            display_value = f"{value} ({mapping[value]})"
                        else:
                            display_value = str(value)

                        input_data.append({
                            '特征': feature_name,
                            '输入值': display_value
                        })

                    input_df = pd.DataFrame(input_data)
                    st.table(input_df)

    with tab2:
        st.subheader("关于此应用")
        st.markdown("""
        ### 应用说明
        这是一个基于OrderedModel（等级Logistic回归）的健康风险评估预测工具。

        ### 模型特征
        模型使用以下10个特征进行预测：

        1. **性别**: 患者性别
        2. **年龄分层**: 年龄分组
        3. **月收入**: 经济状况指标
        4. **文化程度**: 教育水平
        5. **婚姻状况**: 婚姻状态
        6. **查尔森共病指数评估量表**: 共病情况评估
        7. **运动耐量表**: 身体活动能力评估
        8. **简版老年抑郁量表**: 心理健康评估
        9. **简易营养量表MNA-SF**: 营养状况评估
        10. **BMI分层**: 体重指数分类

        ### 使用说明
        1. 在"单个预测"标签页输入特征值
        2. 点击"开始预测"按钮
        3. 查看预测结果和概率分布

        ### 注意事项
        - 预测结果仅供参考，不能替代专业医疗建议
        - 对于重要决策，建议结合临床评估和其他检查结果
        """)


if __name__ == "__main__":
    main()