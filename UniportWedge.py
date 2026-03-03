import streamlit as st
import joblib
import numpy as np
import os

# ==========================================
# 1. 页面基础配置 / Page Configuration
# ==========================================
st.set_page_config(
    page_title="机器学习肺楔形切除术引流管预测 / Machine Learning Prediction of Drainage Tube in Pulmonary Wedge Resection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 新增：添加Logo / Add Logo (在最顶部)
# ==========================================
import base64

current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "logo.png")

if os.path.exists(logo_path):
    # 读取图片并转换为Base64，用于HTML嵌入
    with open(logo_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
    
    # 使用 HTML + CSS 实现图片绝对居中
    # width: 120px 控制图片大小（约4个字宽），可自行调整
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
    st.warning("⚠️ 未找到 logo.png 文件，请确保图片与程序在同一文件夹下。 / Logo.png file not found, please ensure the image is in the same folder as the program.")
    st.markdown("---")
    
# ==========================================
# 标题 / Title
# ==========================================
st.title("🫁 机器学习肺楔形切除术不留置引流管成功概率预测 / Machine Learning Prediction of Success Probability Without Drainage After Pulmonary Wedge Resection")
st.markdown("---")

# ==========================================
# 2. 模型加载 / Model Loading (修改为云端兼容的相对路径 / Modified for Cloud-Compatible Relative Path)
# ==========================================
@st.cache_resource
def load_my_model():
    # --- 修改开始 / Modification Start ---
    # 获取当前脚本所在的文件夹路径 / Get the directory path of the current script
    current_dir = os.path.dirname(__file__)
    # 拼接模型文件名 (确保模型文件名和你实际的文件名完全一致，包括大小写) / Concatenate model filename (Ensure the model filename exactly matches your actual file name, including case sensitivity)
    MODEL_FILENAME = 'LightGBM.joblib'
    MODEL_PATH = os.path.join(current_dir, MODEL_FILENAME)
    # --- 修改结束 / Modification End ---
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 未找到模型文件，请检查文件是否在同一目录下 / Model file not found, please check if the file is in the same directory")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ 模型加载成功 / Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败 / Model loading failed: {str(e)}")
        return None

model = load_my_model()

# ==========================================
# 3. 输入表单 / Input Form (按临床逻辑分组 / Grouped by Clinical Logic)
# ==========================================
with st.form("prediction_form"):
    # --- 第一部分：患者基础信息 / Part 1: Basic Patient Information ---
    st.subheader("👤 一、患者基础信息 / 1. Basic Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("年龄 (Age, 岁 / years)", min_value=0, max_value=120, step=1, value=60)
    with col2:
        # 注意：请确保此处编码(0/1)与你训练模型时完全一致 / Note: Please ensure the coding (0/1) here is exactly the same as when training your model
        Gender = st.radio("性别 (Gender)", options=[0, 1], format_func=lambda x: "女 / Female" if x == 0 else "男 / Male", horizontal=True)
    with col3:
        BMI = st.number_input("体重指数 (BMI, kg/m²)", min_value=10.0, max_value=50.0, step=0.1, value=22.0)

    st.markdown("---")
    
    # --- 第二部分：术前肺功能 / Part 2: Preoperative Pulmonary Function ---
    st.subheader("🫁 二、术前肺功能 / 2. Preoperative Pulmonary Function")
    col1, col2, col3 = st.columns(3)
    with col1:
        FEV1_residual = st.number_input("术前 FEV1 实测值 (Preoperative FEV1 residual, L)", min_value=0.0, step=0.01, value=2.5)
    with col2:
        FEV1_predicted = st.number_input("术前 FEV1 预计值 (%) (Preoperative FEV1 predicted (%))", min_value=0.0, max_value=200.0, step=0.1, value=85.0)
    with col3:
        FEV1_FVC = st.number_input("术前 FEV1/FVC (Preoperative FEV1/FVC)", min_value=0.0, max_value=2.0, step=0.01, value=0.75)

    st.markdown("---")
    
    # --- 第三部分：手术相关指标 / Part 3: Surgery-Related Indicators ---
    st.subheader("🏥 三、手术相关指标 / 3. Surgery-Related Indicators")
    col1, col2 = st.columns(2)
    with col1:
        Lymph_node_sampling = st.radio("淋巴结采样 (Lymph node sampling)", options=[0, 1], format_func=lambda x: "否 / No" if x == 0 else "是 / Yes", horizontal=True)
        Tumor_size = st.number_input("肿瘤大小 (Tumor size, mm)", min_value=0.0, step=0.1, value=10.0)
        Operation_time = st.number_input("手术时间 (Operation time, min)", min_value=0.0, step=1.0, value=60.0)
    with col2:
        Blood_loss = st.number_input("术中失血量 (Intraoperative blood loss, ml)", min_value=0.0, step=1.0, value=10.0)
        Hemo_filament = st.radio("止血材料-纤丝类 (Filament type)", options=[0, 1], format_func=lambda x: "未使用 / Not used" if x == 0 else "使用 / Used", horizontal=True)
        Hemo_membrane = st.radio("止血材料-膜类 (Membrane type)", options=[0, 1], format_func=lambda x: "未使用 / Not used" if x == 0 else "使用 / Used", horizontal=True)

    st.markdown("---")
    
    # --- 第四部分：术前实验室检查 / Part 4: Preoperative Laboratory Tests ---
    st.subheader("🧪 四、术前实验室检查 / 4. Preoperative Laboratory Tests")
    col1, col2, col3 = st.columns(3)
    with col1:
        WBC = st.number_input("白细胞计数 (Preoperative white blood cell count, 10^9/L)", min_value=0.0, step=0.01, value=6.0)
        Hb = st.number_input("血红蛋白 (Preoperative hemoglobin, g/L)", min_value=0.0, step=0.1, value=130.0)
        PLT = st.number_input("血小板 (Preoperative platelets， 10^9/L)", min_value=0.0, step=1.0, value=200.0)
    with col2:
        ALT = st.number_input("谷丙转氨酶 (Preoperative ALT, U/L)", min_value=0.0, step=0.1, value=20.0)
        AST = st.number_input("谷草转氨酶 (Preoperative AST, U/L)", min_value=0.0, step=0.1, value=20.0)
        Total_protein = st.number_input("总蛋白 (Preoperative total protein, g/L)", min_value=0.0, step=0.1, value=70.0)
    with col3:
        Albumin = st.number_input("白蛋白 (Preoperative albumin, g/L)", min_value=0.0, step=0.1, value=40.0)
        K = st.number_input("血钾 (Preoperative potassium, mmol/L)", min_value=0.0, max_value=10.0, step=0.01, value=4.0)
        Ca = st.number_input("血钙 (Preoperative calcium, mmol/L)", min_value=0.0, max_value=5.0, step=0.01, value=2.2)
        Glucose = st.number_input("血糖 (Preoperative glucose, mmol/L)", min_value=0.0, step=0.01, value=5.0)

    st.markdown("---")
    
    # 提交按钮 / Submit Button
    submitted = st.form_submit_button("🔍 计算成功概率 / Calculate Success Probability", use_container_width=True, type="primary")

# ==========================================
# 4. 预测逻辑与结果展示 / Prediction Logic and Result Display (已修复类型错误 / Type Error Fixed)
# ==========================================
if submitted:
    if model is None:
        st.error("请先确保模型已正确加载！ / Please ensure the model is loaded correctly!")
    else:
        # =======================================================
        # ⚠️ 【核心红线】特征顺序必须与训练时完全一致！ / Feature order must be exactly the same as during training!
        # 严格按照你提供的列表顺序排列 / Strictly follow the list order you provided
        # =======================================================
        input_data = np.array([
            Age, Gender, BMI,
            FEV1_residual, FEV1_predicted, FEV1_FVC,
            Lymph_node_sampling, Tumor_size,
            Operation_time, Blood_loss,
            Hemo_filament, Hemo_membrane,
            WBC, Hb, PLT, ALT, AST,
            Total_protein, Albumin,
            K, Ca, Glucose
        ]).reshape(1, -1)

        # 特征数量校验 / Feature Count Verification
        if input_data.shape[1] != model.n_features_in_:
            st.error(f"特征数量不匹配！模型需要 {model.n_features_in_} 个特征，当前输入了 {input_data.shape[1]} 个。 / Feature count mismatch! Model requires {model.n_features_in_} features, but currently input {input_data.shape[1]}.")
        else:
            # 执行预测 (获取概率) / Execute prediction (get probability)
            try:
                prob_array = model.predict_proba(input_data)[:, 1]
                
                # 【关键修复】将 numpy 的 float32 转换为 python 原生浮点 / Convert numpy float32 to native python float
                prob = float(prob_array[0]) 
                prob_percent = prob * 100

                # 可视化结果 / Visualize Results
                st.markdown("## 📊 预测结果 / Prediction Results")
                
                # 分三列展示，中间放大结果 / Display in three columns, enlarge result in the middle
                l_col, m_col, r_col = st.columns([1, 2, 1])
                
                with m_col:
                    # 根据概率高低显示不同颜色 / Display different colors based on probability level
                    if prob_percent >= 50:
                        st.success(f"### 不留置引流管成功概率 / Success Probability Without Drainage Tube")
                        st.markdown(f"<h1 style='text-align: center; color: #0f766e;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                        st.info("💡 提示：成功概率较高，建议结合临床情况综合判断。 / Tip: High success probability, recommended to comprehensively judge based on clinical situation.")
                    else:
                        st.error(f"### 不留置引流管成功概率 / Success Probability Without Drainage Tube")
                        st.markdown(f"<h1 style='text-align: center; color: #dc2626;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                        st.info("💡 提示：成功概率较低，建议做好留置引流管准备。 / Tip: Low success probability, recommend preparing for drainage tube placement.")

                    # 【关键修复】这里传入原生浮点而不是 numpy float32 / Pass native float instead of numpy float32
                    st.progress(prob)

                # =======================================================
                # 仅保留：研究人群和结局定义图片展示
                # Only keep: Study Population and Outcome Definition Image Display
                # =======================================================
                st.markdown("---")
                st.subheader("👥 研究人群和结局定义 / Study Population and Outcome Definition")
                patients_outcome_path = os.path.join(current_dir, "Patients_and_Outcome.png")
                if os.path.exists(patients_outcome_path):
                    _, po_col, _ = st.columns([1, 2, 1])
                    with po_col:
                        st.image(patients_outcome_path, caption="研究人群和结局定义 / Study Population and Outcome Definition", use_container_width=True)
                else:
                    st.warning("⚠️ 未找到研究人群和结局定义图片 / Patients_and_Outcome.png not found.")

            except Exception as e:
                st.error(f"预测过程中发生错误 / Error occurred during prediction：{str(e)}")
