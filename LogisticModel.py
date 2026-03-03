import streamlit as st
import joblib
import numpy as np
import os

# ==========================================
# 1. 页面基础配置 / Page Configuration
# ==========================================
st.set_page_config(
    page_title="机器学习肺楔形切除术引流管预测",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Logo 部分 (保持不变)
# ==========================================
import base64
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "logo.png")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: 120px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
else:
    st.warning("⚠️ 未找到 logo.png")
    st.markdown("---")
    
# ==========================================
# 标题 / Title
# ==========================================
st.title("🫁 机器学习肺楔形切除术不留置引流管成功概率预测")
st.markdown("---")

# ==========================================
# 2. 加载 Model 和 Scaler
# ==========================================
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(current_dir, 'logistic_final_model.joblib')
    SCALER_PATH = os.path.join(current_dir, 'scaler_final.joblib')
    
    model = None
    scaler = None
    
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.success("✅ 模型加载成功")
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
    else:
        st.error("❌ 未找到模型文件")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            st.success("✅ 标准化器加载成功")
        except Exception as e:
            st.error(f"❌ 标准化器加载失败: {str(e)}")
    else:
        st.error("❌ 未找到 scaler_final.joblib")

    return model, scaler

model, scaler = load_models()

# ==========================================
# 3. 输入表单 (保持不变)
# ==========================================
with st.form("prediction_form"):
    st.subheader("👤 一、患者基础信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("年龄 (Age, 岁)", min_value=0, max_value=120, step=1, value=60)
    with col2:
        Gender = st.radio("性别 (Gender)", options=[0, 1], format_func=lambda x: "女" if x == 0 else "男", horizontal=True)
    with col3:
        BMI = st.number_input("体重指数 (BMI, kg/m²)", min_value=10.0, max_value=50.0, step=0.1, value=22.0)

    st.markdown("---")
    
    st.subheader("🫁 二、术前肺功能")
    col1, col2, col3 = st.columns(3)
    with col1:
        FEV1_residual = st.number_input("术前 FEV1 实测值 (L)", min_value=0.0, step=0.01, value=2.5)
    with col2:
        FEV1_predicted = st.number_input("术前 FEV1 预计值 (%)", min_value=0.0, max_value=200.0, step=0.1, value=85.0)
    with col3:
        FEV1_FVC = st.number_input("术前 FEV1/FVC", min_value=0.0, max_value=2.0, step=0.01, value=0.75)

    st.markdown("---")
    
    st.subheader("🏥 三、手术相关指标")
    col1, col2 = st.columns(2)
    with col1:
        Lymph_node_sampling = st.radio("淋巴结采样", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是", horizontal=True)
        Tumor_size = st.number_input("肿瘤大小 (Tumor size, mm)", min_value=0.0, step=0.1, value=10.0)
        Operation_time = st.number_input("手术时间 (Operation time, min)", min_value=0.0, step=1.0, value=60.0)
    with col2:
        Blood_loss = st.number_input("术中失血量 (Intraoperative blood loss, ml)", min_value=0.0, step=1.0, value=10.0)
        Hemo_filament = st.radio("止血材料-纤丝类", options=[0, 1], format_func=lambda x: "未使用" if x == 0 else "使用", horizontal=True)
        Hemo_membrane = st.radio("止血材料-膜类", options=[0, 1], format_func=lambda x: "未使用" if x == 0 else "使用", horizontal=True)

    st.markdown("---")
    
    st.subheader("🧪 四、术前实验室检查")
    col1, col2, col3 = st.columns(3)
    with col1:
        WBC = st.number_input("白细胞计数 (10^9/L)", min_value=0.0, step=0.01, value=6.0)
        Hb = st.number_input("血红蛋白 (g/L)", min_value=0.0, step=0.1, value=130.0)
        PLT = st.number_input("血小板 (10^9/L)", min_value=0.0, step=1.0, value=200.0)
    with col2:
        ALT = st.number_input("谷丙转氨酶 (U/L)", min_value=0.0, step=0.1, value=20.0)
        AST = st.number_input("谷草转氨酶 (U/L)", min_value=0.0, step=0.1, value=20.0)
        Total_protein = st.number_input("总蛋白 (g/L)", min_value=0.0, step=0.1, value=70.0)
    with col3:
        Albumin = st.number_input("白蛋白 (g/L)", min_value=0.0, step=0.1, value=40.0)
        K = st.number_input("血钾 (mmol/L)", min_value=0.0, max_value=10.0, step=0.01, value=4.0)
        Ca = st.number_input("血钙 (mmol/L)", min_value=0.0, max_value=5.0, step=0.01, value=2.2)
        Glucose = st.number_input("血糖 (mmol/L)", min_value=0.0, step=0.01, value=5.0)

    st.markdown("---")
    submitted = st.form_submit_button("🔍 计算成功概率", use_container_width=True, type="primary")

# ==========================================
# 4. 预测逻辑 (标准化逻辑保持不变)
# ==========================================
if submitted:
    if model is None or scaler is None:
        st.error("请先确保模型和标准化器都已正确加载！")
    else:
        try:
            # 步骤 1: 按原始顺序收集所有 22 个变量
            full_features_list = [
                Age, Gender, BMI,
                FEV1_residual, FEV1_predicted, FEV1_FVC,
                Lymph_node_sampling, Tumor_size,
                Operation_time, Blood_loss,
                Hemo_filament, Hemo_membrane,
                WBC, Hb, PLT, ALT, AST,
                Total_protein, Albumin,
                K, Ca, Glucose
            ]
            full_data = np.array(full_features_list).reshape(1, -1)

            # 步骤 2: 定义需要/不需要标准化的列索引
            continuous_indices = [0, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            categorical_indices = [1, 6, 10, 11]

            # 步骤 3: 拆分数据
            data_continuous = full_data[:, continuous_indices]
            data_categorical = full_data[:, categorical_indices]

            # 步骤 4: 执行标准化
            data_continuous_scaled = scaler.transform(data_continuous)

            # 步骤 5: 按顺序拼接回去
            final_input = np.zeros((1, 22))
            final_input[:, continuous_indices] = data_continuous_scaled
            final_input[:, categorical_indices] = data_categorical

            # 步骤 6: 执行预测
            prob_array = model.predict_proba(final_input)[:, 1]
            prob = float(prob_array[0]) 
            prob_percent = prob * 100

            # 可视化预测结果
            st.markdown("## 📊 预测结果")
            l_col, m_col, r_col = st.columns([1, 2, 1])
            
            with m_col:
                if prob_percent >= 50:
                    st.success(f"### 不留置引流管成功概率")
                    st.markdown(f"<h1 style='text-align: center; color: #0f766e;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                    st.info("💡 提示：成功概率较高，建议结合临床情况综合判断。")
                else:
                    st.error(f"### 不留置引流管成功概率")
                    st.markdown(f"<h1 style='text-align: center; color: #dc2626;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                    st.info("💡 提示：成功概率较低，建议做好留置引流管准备。")
                st.progress(prob)
            
            # =======================================================
            # 【核心修改】混淆矩阵改为自适应宽度
            # =======================================================
            st.markdown("---")
            st.subheader("📈 模型性能参考 / Model Performance Reference")
            cm_path = os.path.join(current_dir, "Confusion_Matrix_for_LR.png")
            
            if os.path.exists(cm_path):
                # 三列居中布局 + 自适应宽度
                _, cm_col, _ = st.columns([1, 2, 1])
                with cm_col:
                    # 关键修改：use_container_width=True 实现自适应
                    st.image(cm_path, caption="混淆矩阵 / Confusion Matrix", use_container_width=True)
            else:
                st.warning("⚠️ 未找到混淆矩阵图片 / Confusion matrix image not found.")

        except Exception as e:
            st.error(f"预测过程中发生错误：{str(e)}")
            st.info("如果错误提示特征数量不匹配，请检查代码中 'continuous_indices' 的索引列表是否与训练时的变量筛选逻辑完全一致。")