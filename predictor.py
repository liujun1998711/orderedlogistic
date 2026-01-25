# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import json

# === 设置页面配置 ===
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

# === 页面标题 ===
st.title("📈 等级Logistic回归预测计算器")
st.markdown("""
使用OrderedModel（等级logistic回归）进行健康风险评估预测。请在下方的输入框中输入相关特征值，
系统将自动计算各个等级的概率和最终预测结果。
""")


# === 加载模型函数 ===
@st.cache_resource
def load_model():
    """加载预训练的OrderedModel"""
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        st.sidebar.success("✅ 模型加载成功")

        # 显示模型信息
        st.sidebar.info(f"特征数量: {len(model_data['feature_names'])}")
        if 'n_categories' in model_data:
            st.sidebar.info(f"等级数量: {model_data['n_categories']}")
        elif 'y_categories' in model_data:
            st.sidebar.info(f"等级数量: {len(model_data['y_categories'])}")

        return model_data
    except FileNotFoundError:
        st.error("❌ 模型文件未找到！请确保 'ordered_logit_model.pkl' 存在。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {str(e)}")
        st.stop()


# === 预测函数 ===
def predict_ordered_model(features_dict, model_data):
    """使用OrderedModel进行预测"""
    try:
        # 获取特征名称（确保顺序与训练时一致）
        feature_names = model_data['feature_names']

        # 将用户输入转换为模型需要的格式（按特征顺序）
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
            'feature_values': features_dict,
            'feature_names': feature_names
        }
    except Exception as e:
        st.error(f"预测时出错: {str(e)}")
        return None


# === 主函数 ===
def main():
    # === 加载模型 ===
    model_data = load_model()

    # === 在侧边栏显示版本信息 ===
    with st.sidebar:
        st.header("📋 版本信息")

        # 尝试加载版本信息
        try:
            with open("library_versions.json", "r") as f:
                versions = json.load(f)

            for lib, ver in versions.items():
                st.caption(f"{lib}: {ver}")
        except:
            st.caption("版本信息文件未找到")

        st.header("⚙️ 特征映射说明")

        # 显示特征映射
        if 'feature_mappings' in model_data:
            with st.expander("查看特征编码"):
                for feature, mapping in model_data['feature_mappings'].items():
                    st.write(f"**{feature}**:")
                    for code, label in mapping.items():
                        st.write(f"  {code}: {label}")
        else:
            st.warning("模型中没有特征映射信息")

    # === 创建标签页 ===
    tab1, tab2, tab3 = st.tabs(["🔍 单个预测", "📊 批量预测", "ℹ️ 关于"])

    with tab1:
        st.subheader("单个样本预测")

        # === 创建特征输入区域 ===
        features_dict = {}

        # 创建两列布局
        col1, col2 = st.columns(2)

        with col1:
            # 1. 性别
            if 'feature_mappings' in model_data and '性别' in model_data['feature_mappings']:
                gender_mapping = model_data['feature_mappings']['性别']
                gender_options = list(gender_mapping.keys())
                gender_labels = [gender_mapping[code] for code in gender_options]

                gender_label = st.selectbox(
                    "性别",
                    options=gender_options,
                    format_func=lambda x: gender_mapping[x],
                    help="请选择性别"
                )
                features_dict['性别'] = gender_label
            else:
                gender_label = st.selectbox(
                    "性别",
                    options=[0, 1],
                    format_func=lambda x: "女" if x == 0 else "男",
                    help="请选择性别"
                )
                features_dict['性别'] = gender_label

            # 2. 年龄分层
            if 'feature_mappings' in model_data and '年龄分层' in model_data['feature_mappings']:
                age_mapping = model_data['feature_mappings']['年龄分层']
                age_options = list(age_mapping.keys())
                age_labels = [age_mapping[code] for code in age_options]

                age_label = st.selectbox(
                    "年龄分层",
                    options=age_options,
                    format_func=lambda x: age_mapping[x],
                    help="请选择年龄分层"
                )
                features_dict['年龄分层'] = age_label
            else:
                age_label = st.selectbox(
                    "年龄分层",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'}[x],
                    help="请选择年龄分层"
                )
                features_dict['年龄分层'] = age_label

            # 3. 月收入
            if 'feature_mappings' in model_data and '月收入' in model_data['feature_mappings']:
                income_mapping = model_data['feature_mappings']['月收入']
                income_options = list(income_mapping.keys())
                income_labels = [income_mapping[code] for code in income_options]

                income_label = st.selectbox(
                    "月收入",
                    options=income_options,
                    format_func=lambda x: income_mapping[x],
                    help="请选择月收入范围"
                )
                features_dict['月收入'] = income_label
            else:
                income_label = st.selectbox(
                    "月收入",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'}[x],
                    help="请选择月收入范围"
                )
                features_dict['月收入'] = income_label

            # 4. 文化程度
            if 'feature_mappings' in model_data and '文化程度' in model_data['feature_mappings']:
                edu_mapping = model_data['feature_mappings']['文化程度']
                edu_options = list(edu_mapping.keys())
                edu_labels = [edu_mapping[code] for code in edu_options]

                edu_label = st.selectbox(
                    "文化程度",
                    options=edu_options,
                    format_func=lambda x: edu_mapping[x],
                    help="请选择文化程度"
                )
                features_dict['文化程度'] = edu_label
            else:
                edu_label = st.selectbox(
                    "文化程度",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'}[x],
                    help="请选择文化程度"
                )
                features_dict['文化程度'] = edu_label

            # 5. 婚姻状况
            if 'feature_mappings' in model_data and '婚姻状况' in model_data['feature_mappings']:
                marital_mapping = model_data['feature_mappings']['婚姻状况']
                marital_options = list(marital_mapping.keys())
                marital_labels = [marital_mapping[code] for code in marital_options]

                marital_label = st.selectbox(
                    "婚姻状况",
                    options=marital_options,
                    format_func=lambda x: marital_mapping[x],
                    help="请选择婚姻状况"
                )
                features_dict['婚姻状况'] = marital_label
            else:
                marital_label = st.selectbox(
                    "婚姻状况",
                    options=[0, 1],
                    format_func=lambda x: {0: '已婚', 1: '离异/丧偶'}[x],
                    help="请选择婚姻状况"
                )
                features_dict['婚姻状况'] = marital_label

        with col2:
            # 6. 查尔森共病指数评估量表 (连续变量)
            charlson = st.slider(
                "查尔森共病指数评估量表",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="查尔森共病指数评分，范围0-20"
            )
            features_dict['查尔森共病指数评估量表'] = charlson

            # 7. 运动耐量表 (连续变量)
            exercise = st.slider(
                "运动耐量表",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="运动耐量评分，范围0-10"
            )
            features_dict['运动耐量表'] = exercise

            # 8. 简版老年抑郁量表 (连续变量)
            depression = st.slider(
                "简版老年抑郁量表",
                min_value=0,
                max_value=15,
                value=5,
                step=1,
                help="老年抑郁量表评分，范围0-15"
            )
            features_dict['简版老年抑郁量表'] = depression

            # 9. 简易营养量表MNA-SF (连续变量)
            mna_sf = st.slider(
                "简易营养量表MNA-SF",
                min_value=0,
                max_value=14,
                value=8,
                step=1,
                help="营养状况评分，范围0-14"
            )
            features_dict['简易营养量表MNA-SF'] = mna_sf

            # 10. BMI分层
            if 'feature_mappings' in model_data and 'BMI分层' in model_data['feature_mappings']:
                bmi_mapping = model_data['feature_mappings']['BMI分层']
                bmi_options = list(bmi_mapping.keys())
                bmi_labels = [bmi_mapping[code] for code in bmi_options]

                bmi_label = st.selectbox(
                    "BMI分层",
                    options=bmi_options,
                    format_func=lambda x: bmi_mapping[x],
                    help="请选择BMI分层"
                )
                features_dict['BMI分层'] = bmi_label
            else:
                bmi_label = st.selectbox(
                    "BMI分层",
                    options=[4, 3, 2, 1],  # 注意顺序：4:偏瘦, 3:正常, 2:超重, 1:肥胖
                    format_func=lambda x: {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}[x],
                    help="请选择BMI分层"
                )
                features_dict['BMI分层'] = bmi_label

        # === 预测按钮 ===
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                # 验证特征数量
                if len(features_dict) != len(model_data['feature_names']):
                    st.error(f"特征数量不匹配！期望 {len(model_data['feature_names'])}，得到 {len(features_dict)}")
                    return

                # 进行预测
                result = predict_ordered_model(features_dict, model_data)

                if result is None:
                    return

                # === 显示预测结果 ===
                st.success("✅ 预测完成！")

                # 创建结果展示区域
                st.subheader("📋 预测结果")

                # 显示预测等级（卡片样式）
                col_result1, col_result2 = st.columns([1, 2])

                with col_result1:
                    # 使用HTML创建漂亮的卡片
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        margin-bottom: 1rem;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    '>
                        <h3 style='color: white; margin: 0; font-size: 1.2rem;'>预测等级</h3>
                        <h1 style='font-size: 3rem; margin: 0.5rem 0;'>{result['prediction_label']}</h1>
                        <p style='margin: 0; font-size: 1rem;'>置信度: {result['probabilities'][result['prediction']]:.2%}</p>
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
                            return ['background-color: #e6f7ff', 'background-color: #e6f7ff']
                        return ['', '']

                    styled_df = prob_df[['等级', '概率 (%)']].style.apply(highlight_row, axis=1)
                    st.dataframe(styled_df, use_container_width=True)

                # === 概率可视化 ===
                st.subheader("📈 等级概率分布图")

                # 创建柱状图
                fig = go.Figure(data=[
                    go.Bar(
                        x=result['class_labels'],
                        y=result['probabilities'],
                        marker_color=['#FF6B6B' if i == result['prediction'] else '#4ECDC4'
                                      for i in range(len(result['class_labels']))],
                        text=[f'{p:.1%}' for p in result['probabilities']],
                        textposition='auto',
                        hoverinfo='x+y',
                        hovertemplate='等级: %{x}<br>概率: %{y:.2%}<extra></extra>'
                    )
                ])

                fig.update_layout(
                    title="各等级预测概率分布",
                    xaxis_title="等级",
                    yaxis_title="概率",
                    yaxis=dict(tickformat=".0%"),
                    height=400,
                    showlegend=False,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

                # === 累积概率图 ===
                st.subheader("📊 累积概率分布")

                cumulative_probs = np.cumsum(result['probabilities'])

                fig2 = go.Figure(data=[
                    go.Scatter(
                        x=result['class_labels'],
                        y=cumulative_probs,
                        mode='lines+markers+text',
                        line=dict(width=3, color='#36A2EB'),
                        marker=dict(size=10, color='#FF6384'),
                        text=[f'{p:.0%}' for p in cumulative_probs],
                        textposition="top center",
                        fill='tozeroy',
                        fillcolor='rgba(54, 162, 235, 0.2)'
                    )
                ])

                fig2.update_layout(
                    title="累积概率分布",
                    xaxis_title="等级",
                    yaxis_title="累积概率",
                    yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                    height=300,
                    template="plotly_white"
                )

                st.plotly_chart(fig2, use_container_width=True)

                # === 显示输入的特征值 ===
                with st.expander("📝 查看输入的特征值"):
                    # 创建特征值表格
                    input_data = []
                    for feature_name in result['feature_names']:
                        value = result['feature_values'][feature_name]

                        # 如果有特征映射，显示标签
                        if 'feature_mappings' in model_data and feature_name in model_data['feature_mappings']:
                            mapping = model_data['feature_mappings'][feature_name]
                            if value in mapping:
                                display_value = f"{value} ({mapping[value]})"
                            else:
                                display_value = str(value)
                        else:
                            display_value = str(value)

                        input_data.append({
                            '特征': feature_name,
                            '输入值': display_value
                        })

                    input_df = pd.DataFrame(input_data)
                    st.table(input_df)

                # === 模型解释和建议 ===
                with st.expander("💡 结果解释和建议"):
                    st.write("""
                    ### 结果解释：
                    1. **预测等级**：模型认为最可能的结果
                    2. **置信度**：模型对预测结果的把握程度
                    3. **概率分布**：各个可能等级的概率

                    ### 建议：
                    - **高置信度(>70%)**：结果较为可靠
                    - **中等置信度(40%-70%)**：结果有一定参考价值，建议结合其他评估
                    - **低置信度(<40%)**：结果不确定性较高，建议重新评估或咨询专家
                    """)

    with tab2:
        st.subheader("批量预测")

        # 文件上传
        uploaded_file = st.file_uploader(
            "上传CSV文件进行批量预测",
            type=['csv'],
            help="请确保CSV文件包含所有必需的特征列"
        )

        if uploaded_file:
            try:
                # 读取CSV文件
                batch_df = pd.read_csv(uploaded_file)

                # 检查是否包含所有必需特征
                required_features = model_data['feature_names']
                missing_features = [f for f in required_features if f not in batch_df.columns]

                if missing_features:
                    st.error(f"缺少以下特征列: {', '.join(missing_features)}")
                    st.info(f"需要的特征列: {', '.join(required_features)}")
                else:
                    # 确保列顺序正确
                    batch_df = batch_df[required_features]

                    # 批量预测
                    predictions = []
                    probabilities_list = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, (_, row) in enumerate(batch_df.iterrows()):
                        # 更新进度
                        progress = (i + 1) / len(batch_df)
                        progress_bar.progress(progress)
                        status_text.text(f"正在处理第 {i + 1}/{len(batch_df)} 条记录...")

                        # 转换为字典
                        row_dict = dict(zip(required_features, row.tolist()))
                        result = predict_ordered_model(row_dict, model_data)

                        if result:
                            predictions.append(result['prediction_label'])
                            probabilities_list.append(result['probabilities'])
                        else:
                            predictions.append(None)
                            probabilities_list.append(
                                [None] * len(result['class_labels']) if 'result' in locals() else [None])

                    progress_bar.empty()
                    status_text.empty()

                    # 添加预测结果到原始数据
                    result_df = batch_df.copy()
                    result_df['预测等级'] = predictions

                    # 添加每个等级的概率
                    if probabilities_list and probabilities_list[0] is not None:
                        for i in range(len(probabilities_list[0])):
                            result_df[f'等级{i}_概率'] = [probs[i] if probs is not None else None
                                                          for probs in probabilities_list]

                    # 显示结果
                    st.dataframe(result_df.head(20), use_container_width=True)

                    # 统计信息
                    st.subheader("📊 批量预测统计")

                    if predictions:
                        # 计算统计量
                        prediction_counts = pd.Series(predictions).value_counts()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总样本数", len(result_df))
                        with col2:
                            valid_probs = [max(probs) for probs in probabilities_list if
                                           probs is not None and None not in probs]
                            if valid_probs:
                                avg_prob = np.mean(valid_probs)
                                st.metric("平均最高概率", f"{avg_prob:.1%}")
                            else:
                                st.metric("平均最高概率", "N/A")
                        with col3:
                            if not prediction_counts.empty:
                                most_common = prediction_counts.index[0]
                                st.metric("最常见等级", most_common)
                            else:
                                st.metric("最常见等级", "N/A")

                        # 等级分布饼图
                        if not prediction_counts.empty:
                            fig3 = px.pie(
                                values=prediction_counts.values,
                                names=prediction_counts.index,
                                title='预测等级分布',
                                hole=0.3,
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            fig3.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig3, use_container_width=True)

                    # 下载按钮
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载预测结果",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")

        else:
            st.info("👆 请上传CSV文件进行批量预测")

            # 提供模板下载
            template_df = pd.DataFrame(columns=model_data['feature_names'])
            template_csv = template_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📋 下载数据模板",
                data=template_csv,
                file_name="prediction_template.csv",
                mime="text/csv"
            )

    with tab3:
        st.subheader("关于此应用")
        st.markdown("""
        ### 应用说明
        这是一个基于OrderedModel（等级Logistic回归）的健康风险评估预测工具。

        ### 模型特征
        模型使用以下10个特征进行预测：

        1. **性别**：患者性别
        2. **年龄分层**：年龄分组
        3. **月收入**：经济状况指标
        4. **文化程度**：教育水平
        5. **婚姻状况**：婚姻状态
        6. **查尔森共病指数评估量表**：共病情况评估
        7. **运动耐量表**：身体活动能力评估
        8. **简版老年抑郁量表**：心理健康评估
        9. **简易营养量表MNA-SF**：营养状况评估
        10. **BMI分层**：体重指数分类

        ### 使用说明
        1. **单个预测**：在"单个预测"标签页输入特征值，点击"开始预测"按钮
        2. **批量预测**：在"批量预测"标签页上传CSV文件进行批量预测
        3. **结果解释**：查看预测结果、概率分布和累积概率

        ### 注意事项
        - 请确保输入的特征值与训练模型时使用的编码一致
        - 预测结果仅供参考，不能替代专业医疗建议
        - 对于重要决策，建议结合临床评估和其他检查结果
        """)

        # 显示模型技术信息
        with st.expander("技术信息"):
            st.write("**模型类型**: OrderedModel (等级Logistic回归)")
            st.write("**分布**: Logit")
            st.write("**特征数量**: 10")

            if 'training_info' in model_data:
                st.write("**训练信息**:")
                for key, value in model_data['training_info'].items():
                    st.write(f"  - {key}: {value}")

    # === 页脚 ===
    st.markdown("---")
    st.caption("© 2024 健康风险评估预测系统 | 基于OrderedModel构建")


# === 运行应用 ===
if __name__ == "__main__":
    main()