import streamlit as st
import pickle
import pandas as pd
import numpy as np
import traceback

# 设置页面配置
st.set_page_config(
    page_title="等级Logistic回归预测计算器",
    page_icon="📊",
    layout="wide"
)

st.title("📈 等级Logistic回归预测计算器")
st.markdown("使用OrderedModel（等级logistic回归）进行健康风险评估预测")

@st.cache_resource
def load_model():
    try:
        with open('ordered_logit_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

def predict_single(features_dict, model_data):
    try:
        feature_names = model_data['feature_names']
        features_list = []
        for f in feature_names:
            val = features_dict.get(f)
            if val is None:
                raise KeyError(f"缺少特征: {f}")
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            features_list.append(val)
        
        X_new = pd.DataFrame([features_list], columns=feature_names)
        res = model_data['res']
        pred_raw = res.predict(X_new)
        probs = np.asarray(pred_raw, dtype=np.float64).flatten()
        probabilities = probs.tolist()
        predicted_class = int(np.argmax(probabilities))
        
        class_labels = model_data.get('y_category_labels')
        if class_labels is None:
            class_labels = model_data.get('y_categories')
        if class_labels is None:
            class_labels = [f"等级{i}" for i in range(len(probabilities))]
        else:
            class_labels = list(class_labels)
            if len(class_labels) != len(probabilities):
                class_labels = [f"等级{i}" for i in range(len(probabilities))]
        
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
            'class_labels': class_labels,
            'feature_values': features_dict
        }
    except Exception as e:
        st.error(f"预测时出错: {str(e)}")
        st.code(traceback.format_exc(), language="python")
        return None

def main():
    model_data = load_model()
    if model_data is None:
        st.stop()
    
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
        features_dict = {}
        
        with col1:
            features_dict['Gender'] = st.selectbox("性别", [0,1], format_func=lambda x: "女" if x==0 else "男")
            features_dict['Age_Group'] = st.selectbox("年龄分层", [1,2,3], format_func=lambda x: ["60-69岁","70-79岁","≥80岁"][x-1])
            features_dict['Monthly_Income'] = st.selectbox("月收入", [1,2,3,4], format_func=lambda x: ["<2000元","2000-4000元","4000-6000元",">6000元"][x-1])
            features_dict['Education_Level'] = st.selectbox("文化程度", [1,2,3,4], format_func=lambda x: ["小学及以下","初中","高中/中专","大专及以上"][x-1])
            features_dict['Marital_Status'] = st.selectbox("婚姻状况", [0,1], format_func=lambda x: ["已婚","离异/丧偶"][x])
            features_dict['Neoadjuvant_Chemoradiotherapy'] = st.selectbox("新辅助放化疗", [1,0], format_func=lambda x: "是" if x==1 else "否")
            features_dict['Tumor_Location'] = st.selectbox("肿瘤部位", [1,0], format_func=lambda x: "累计直肠或直肠恶性肿瘤" if x==1 else "结肠恶性肿瘤")
        
        with col2:
            features_dict['Charlson_Comorbidity_Index'] = st.slider("查尔森共病指数评估量表", 0,20,0)
            features_dict['Exercise_Tolerance_Scale'] = st.selectbox("运动耐量表", [0, 1, 2], format_func=lambda x: f"{x} 级")
            features_dict['GDS_Short_Form'] = st.slider("简版老年抑郁量表", 0,15,5)
            features_dict['MNA_SF'] = st.slider("简易营养量表MNA-SF", 0,14,8)
            features_dict['BMI_Category'] = st.selectbox("BMI分层", [4,3,2,1], format_func=lambda x: ["偏瘦(<18.5)","正常(18.5-24)","超重(24-28)","肥胖(≥28)"][4-x])
            features_dict['Stage'] = st.selectbox("肿瘤分期", [0,1,2,3,4], format_func=lambda x: f"{x}期" if x!=0 else "0期")
            features_dict['ASA_Classification'] = st.selectbox("ASA分级", [2,3,4], format_func=lambda x: f"{x}级")
        
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                if len(features_dict) != len(model_data.get('feature_names', [])):
                    st.error("特征数量不匹配！")
                    return
                result = predict_single(features_dict, model_data)
                if result is None:
                    return
                st.success("✅ 预测完成！")
                st.subheader("📋 预测结果")
                col_result1, col_result2 = st.columns([1,2])
                with col_result1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding:2rem; border-radius:10px; color:white; text-align:center;">
                        <h3 style="color:white; margin:0;">预测等级</h3>
                        <h1 style="font-size:3rem; margin:0.5rem 0;">{result['prediction_label']}</h1>
                        <p style="margin:0;">置信度: {result['probabilities'][result['prediction']]:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_result2:
                    prob_df = pd.DataFrame({'等级': result['class_labels'], '概率': result['probabilities']})
                    prob_df['概率 (%)'] = (prob_df['概率']*100).round(2)
                    st.dataframe(prob_df.style.apply(lambda row: ['background-color: #e6f7ff' if row.name==result['prediction'] else '' for _ in row], axis=1))
    
    with tab2:
        st.subheader("关于此应用")
        st.markdown("""
        ### 应用说明
        基于OrderedModel的等级Logistic回归预测工具。
        ### 模型特征（14个）
        ...（您原有的内容）
        """)

if __name__ == "__main__":
    main()
