# app.py - 完整应用（界面中文，特征键英文）
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

# 预测函数（特征键为英文）
def predict_single(features_dict, model_data):
    """单个样本预测，features_dict的键必须是英文"""
    try:
        feature_names = model_data['feature_names']
        features_list = []
        for feature in feature_names:
            if feature not in features_dict:
                raise KeyError(f"缺少特征: {feature}")
            features_list.append(features_dict[feature])
        
        X_new = pd.DataFrame([features_list], columns=feature_names)
        res = model_data['res']
        predicted_probs = res.predict(X_new)
        
        if predicted_probs.ndim == 2:
            probabilities = predicted_probs[0]
        else:
            probabilities = predicted_probs
        
        predicted_class = np.argmax(probabilities)
        
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
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    # 侧边栏模型信息
    st.sidebar.success("✅ 模型加载成功")
    st.sidebar.info(f"特征数量: {len(model_data.get('feature_names', []))}")
    with st.sidebar.expander("🔍 模型特征名称（英文）"):
        for i, name in enumerate(model_data.get('feature_names', [])):
            st.write(f"{i+1}. {name}")
    
    tab1, tab2 = st.tabs(["🔍 单个预测", "ℹ️ 关于"])
    
    with tab1:
        st.subheader("单个样本预测")
        st.markdown("### 请输入特征值:")
        
        col1, col2 = st.columns(2)
        
        # 使用英文键存储，但显示中文标签
        features_dict = {}
        
        with col1:
            # 1. 性别 (Gender)
            gender = st.selectbox(
                "性别",
                options=[0, 1],
                format_func=lambda x: "女" if x == 0 else "男",
                help="0: 女, 1: 男"
            )
            features_dict['Gender'] = gender
            
            # 2. 年龄分层 (Age_Group)
            age_group = st.selectbox(
                "年龄分层",
                options=[1, 2, 3],
                format_func=lambda x: ["60-69岁", "70-79岁", "≥80岁"][x-1],
                help="1:60-69岁, 2:70-79岁, 3:≥80岁"
            )
            features_dict['Age_Group'] = age_group
            
            # 3. 月收入 (Monthly_Income)
            monthly_income = st.selectbox(
                "月收入",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["<2000元", "2000-4000元", "4000-6000元", ">6000元"][x-1],
                help="月收入分层"
            )
            features_dict['Monthly_Income'] = monthly_income
            
            # 4. 文化程度 (Education_Level)
            education_level = st.selectbox(
                "文化程度",
                options=[1, 2, 3, 4],
                format_func=lambda x: ["小学及以下", "初中", "高中/中专", "大专及以上"][x-1],
                help="文化程度"
            )
            features_dict['Education_Level'] = education_level
            
            # 5. 婚姻状况 (Marital_Status)
            marital_status = st.selectbox(
                "婚姻状况",
                options=[0, 1],
                format_func=lambda x: ["已婚", "离异/丧偶"][x],
                help="婚姻状况"
            )
            features_dict['Marital_Status'] = marital_status
            
            # 6. 新辅助放化疗 (Neoadjuvant_Chemoradiotherapy)
            neoadjuvant = st.selectbox(
                "新辅助放化疗",
                options=[1, 0],
                format_func=lambda x: "是" if x == 1 else "否",
                help="1: 是, 0: 否"
            )
            features_dict['Neoadjuvant_Chemoradiotherapy'] = neoadjuvant
            
            # 7. 肿瘤部位 (Tumor_Location)
            tumor_location = st.selectbox(
                "肿瘤部位",
                options=[1, 0],
                format_func=lambda x: "累计直肠或直肠恶性肿瘤" if x == 1 else "结肠恶性肿瘤",
                help="1: 累计直肠或直肠恶性肿瘤, 0: 结肠恶性肿瘤"
            )
            features_dict['Tumor_Location'] = tumor_location
        
        with col2:
            # 8. 查尔森共病指数 (Charlson_Comorbidity_Index)
            charlson = st.slider(
                "查尔森共病指数评估量表",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="查尔森共病指数评分"
            )
            features_dict['Charlson_Comorbidity_Index'] = charlson
            
            # 9. 运动耐量表 (Exercise_Tolerance_Scale)
            exercise = st.slider(
                "运动耐量表",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="运动耐量评分"
            )
            features_dict['Exercise_Tolerance_Scale'] = exercise
            
            # 10. 简版老年抑郁量表 (GDS_Short_Form)
            gds = st.slider(
                "简版老年抑郁量表",
                min_value=0,
                max_value=15,
                value=5,
                step=1,
                help="老年抑郁量表评分"
            )
            features_dict['GDS_Short_Form'] = gds
            
            # 11. 简易营养量表 (MNA_SF)
            mna = st.slider(
                "简易营养量表MNA-SF",
                min_value=0,
                max_value=14,
                value=8,
                step=1,
                help="营养状况评分"
            )
            features_dict['MNA_SF'] = mna
            
            # 12. BMI分层 (BMI_Category)
            bmi = st.selectbox(
                "BMI分层",
                options=[4, 3, 2, 1],
                format_func=lambda x: ["偏瘦(<18.5)", "正常(18.5-24)", "超重(24-28)", "肥胖(≥28)"][4-x],
                help="BMI分层"
            )
            features_dict['BMI_Category'] = bmi
            
            # 13. 肿瘤分期 (Stage)
            stage = st.selectbox(
                "肿瘤分期",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: f"{x}期" if x != 0 else "0期",
                help="0: 0期, 1: 1期, 2: 2期, 3: 3期, 4: 4期"
            )
            features_dict['Stage'] = stage
            
            # 14. ASA分级 (ASA_Classification)
            asa = st.selectbox(
                "ASA分级",
                options=[2, 3, 4],
                format_func=lambda x: f"{x}级",
                help="2: 2级, 3: 3级, 4: 4级"
            )
            features_dict['ASA_Classification'] = asa
        
        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                expected_count = len(model_data.get('feature_names', []))
                if len(features_dict) != expected_count:
                    st.error(f"特征数量不匹配！期望 {expected_count} 个，实际输入 {len(features_dict)} 个。")
                    return
                
                result = predict_single(features_dict, model_data)
                if result is None:
                    return
                
                st.success("✅ 预测完成！")
                st.subheader("📋 预测结果")
                
                col_result1, col_result2 = st.columns([1, 2])
                with col_result1:
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
                    prob_df = pd.DataFrame({
                        '等级': result['class_labels'],
                        '概率': result['probabilities']
                    })
                    prob_df['概率 (%)'] = (prob_df['概率'] * 100).round(2)
                    def highlight_row(row):
                        if row.name == result['prediction']:
                            return ['background-color: #e6f7ff'] * len(row)
                        return [''] * len(row)
                    st.dataframe(
                        prob_df[['等级', '概率 (%)']].style.apply(highlight_row, axis=1),
                        use_container_width=True
                    )
                
                # 显示输入特征值（中文显示）
                with st.expander("📝 查看输入的特征值"):
                    # 建立英文到中文的映射
                    chinese_names = {
                        'Gender': '性别',
                        'Age_Group': '年龄分层',
                        'Monthly_Income': '月收入',
                        'Education_Level': '文化程度',
                        'Marital_Status': '婚姻状况',
                        'Charlson_Comorbidity_Index': '查尔森共病指数评估量表',
                        'Exercise_Tolerance_Scale': '运动耐量表',
                        'GDS_Short_Form': '简版老年抑郁量表',
                        'MNA_SF': '简易营养量表MNA-SF',
                        'BMI_Category': 'BMI分层',
                        'Neoadjuvant_Chemoradiotherapy': '新辅助放化疗',
                        'Tumor_Location': '肿瘤部位',
                        'Stage': '肿瘤分期',
                        'ASA_Classification': 'ASA分级'
                    }
                    input_data = []
                    for eng_name, value in features_dict.items():
                        chn_name = chinese_names.get(eng_name, eng_name)
                        # 显示格式化值
                        if eng_name == 'Gender':
                            display_value = f"{value} ({'女' if value == 0 else '男'})"
                        elif eng_name == 'Age_Group':
                            mapping = {1: '60-69岁', 2: '70-79岁', 3: '≥80岁'}
                            display_value = f"{value} ({mapping[value]})"
                        elif eng_name == 'Monthly_Income':
                            mapping = {1: '<2000元', 2: '2000-4000元', 3: '4000-6000元', 4: '>6000元'}
                            display_value = f"{value} ({mapping[value]})"
                        elif eng_name == 'Education_Level':
                            mapping = {1: '小学及以下', 2: '初中', 3: '高中/中专', 4: '大专及以上'}
                            display_value = f"{value} ({mapping[value]})"
                        elif eng_name == 'Marital_Status':
                            mapping = {0: '已婚', 1: '离异/丧偶'}
                            display_value = f"{value} ({mapping[value]})"
                        elif eng_name == 'BMI_Category':
                            mapping = {4: '偏瘦(<18.5)', 3: '正常(18.5-24)', 2: '超重(24-28)', 1: '肥胖(≥28)'}
                            display_value = f"{value} ({mapping[value]})"
                        elif eng_name == 'Neoadjuvant_Chemoradiotherapy':
                            display_value = f"{value} ({'是' if value == 1 else '否'})"
                        elif eng_name == 'Tumor_Location':
                            display_value = f"{value} ({'累计直肠或直肠恶性肿瘤' if value == 1 else '结肠恶性肿瘤'})"
                        elif eng_name == 'Stage':
                            display_value = f"{value}期"
                        elif eng_name == 'ASA_Classification':
                            display_value = f"{value}级"
                        else:
                            display_value = str(value)
                        input_data.append({
                            '特征': chn_name,
                            '输入值': display_value
                        })
                    input_df = pd.DataFrame(input_data)
                    st.table(input_df)
    
    with tab2:
        st.subheader("关于此应用")
        st.markdown("""
        ### 应用说明
        这是一个基于OrderedModel（等级Logistic回归）的健康风险评估预测工具。
        
        ### 模型特征（14个）
        1. **性别** (Gender)
        2. **年龄分层** (Age_Group)
        3. **月收入** (Monthly_Income)
        4. **文化程度** (Education_Level)
        5. **婚姻状况** (Marital_Status)
        6. **新辅助放化疗** (Neoadjuvant_Chemoradiotherapy)
        7. **肿瘤部位** (Tumor_Location)
        8. **查尔森共病指数评估量表** (Charlson_Comorbidity_Index)
        9. **运动耐量表** (Exercise_Tolerance_Scale)
        10. **简版老年抑郁量表** (GDS_Short_Form)
        11. **简易营养量表MNA-SF** (MNA_SF)
        12. **BMI分层** (BMI_Category)
        13. **肿瘤分期** (Stage)
        14. **ASA分级** (ASA_Classification)
        
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
