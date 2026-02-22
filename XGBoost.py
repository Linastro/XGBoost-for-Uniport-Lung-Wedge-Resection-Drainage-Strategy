import streamlit as st
import joblib
import numpy as np
import os

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="肺楔形切除术引流管预测",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 标题
st.title("🫁 肺楔形切除术不留置引流管成功概率预测")
st.markdown("---")

# ==========================================
# 2. 模型加载 (使用缓存，避免重复加载)
# ==========================================
@st.cache_resource
def load_my_model():
    # 请确保此处路径与你本地模型路径完全一致
    MODEL_PATH = '/Users/linastro/Documents/LinDocuments/LCT code/Python/2025.3无引流管楔形回顾研究/数据分析/xgboost_final_model.joblib'
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 未找到模型文件，请检查路径：\n{MODEL_PATH}")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ 模型加载成功 (期望特征数: {model.n_features_in_})")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

model = load_my_model()

# ==========================================
# 3. 输入表单 (按临床逻辑分组)
# ==========================================
with st.form("prediction_form"):
    # --- 第一部分：患者基础信息 ---
    st.subheader("👤 一、患者基础信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("年龄 (Age, 岁)", min_value=0, max_value=120, step=1, value=60)
    with col2:
        # 注意：请确保此处编码(0/1)与你训练模型时完全一致
        Gender = st.radio("性别 (Gender)", options=[0, 1], format_func=lambda x: "女" if x == 0 else "男", horizontal=True)
    with col3:
        BMI = st.number_input("体重指数 (BMI, kg/m²)", min_value=10.0, max_value=50.0, step=0.1, value=22.0)

    st.markdown("---")
    
    # --- 第二部分：术前肺功能 ---
    st.subheader("🫁 二、术前肺功能")
    col1, col2, col3 = st.columns(3)
    with col1:
        FEV1_residual = st.number_input("术前 FEV1 实测值 (Preoperative FEV1 residual)", min_value=0.0, step=0.01, value=2.5)
    with col2:
        FEV1_predicted = st.number_input("术前 FEV1 预计值 (%) (Preoperative FEV1 predicted (%))", min_value=0.0, max_value=200.0, step=0.1, value=85.0)
    with col3:
        FEV1_FVC = st.number_input("术前 FEV1/FVC", min_value=0.0, max_value=2.0, step=0.01, value=0.75)

    st.markdown("---")
    
    # --- 第三部分：手术相关指标 ---
    st.subheader("🏥 三、手术相关指标")
    col1, col2 = st.columns(2)
    with col1:
        Lymph_node_sampling = st.radio("淋巴结采样 (Lymph node sampling)", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是", horizontal=True)
        Tumor_size = st.number_input("肿瘤大小 (Tumor size, mm)", min_value=0.0, step=0.1, value=10.0)
        Operation_time = st.number_input("手术时间 (Operation time, min)", min_value=0.0, step=1.0, value=60.0)
    with col2:
        Blood_loss = st.number_input("术中失血量 (Intraoperative blood loss, ml)", min_value=0.0, step=1.0, value=10.0)
        Hemo_filament = st.radio("止血材料-丝状 (Filament type)", options=[0, 1], format_func=lambda x: "未使用" if x == 0 else "使用", horizontal=True)
        Hemo_membrane = st.radio("止血材料-膜状 (Membrane type)", options=[0, 1], format_func=lambda x: "未使用" if x == 0 else "使用", horizontal=True)

    st.markdown("---")
    
    # --- 第四部分：术前实验室检查 ---
    st.subheader("🧪 四、术前实验室检查")
    col1, col2, col3 = st.columns(3)
    with col1:
        WBC = st.number_input("白细胞计数 (Preoperative white blood cell count)", min_value=0.0, step=0.01, value=6.0)
        Hb = st.number_input("血红蛋白 (Preoperative hemoglobin)", min_value=0.0, step=0.1, value=130.0)
        PLT = st.number_input("血小板 (Preoperative platelets)", min_value=0.0, step=1.0, value=200.0)
    with col2:
        ALT = st.number_input("谷丙转氨酶 (Preoperative ALT)", min_value=0.0, step=0.1, value=20.0)
        AST = st.number_input("谷草转氨酶 (Preoperative AST)", min_value=0.0, step=0.1, value=20.0)
        Total_protein = st.number_input("总蛋白 (Preoperative total protein)", min_value=0.0, step=0.1, value=70.0)
    with col3:
        Albumin = st.number_input("白蛋白 (Preoperative albumin)", min_value=0.0, step=0.1, value=40.0)
        K = st.number_input("血钾 (Preoperative potassium)", min_value=0.0, max_value=10.0, step=0.01, value=4.0)
        Ca = st.number_input("血钙 (Preoperative calcium)", min_value=0.0, max_value=5.0, step=0.01, value=2.2)
        Glucose = st.number_input("血糖 (Preoperative glucose)", min_value=0.0, step=0.01, value=5.0)

    st.markdown("---")
    
    # 提交按钮
    submitted = st.form_submit_button("🔍 计算成功概率", use_container_width=True, type="primary")

# ==========================================
# 4. 预测逻辑与结果展示 (已修复类型错误)
# ==========================================
if submitted:
    if model is None:
        st.error("请先确保模型已正确加载！")
    else:
        # =======================================================
        # ⚠️ 【核心红线】特征顺序必须与训练时完全一致！
        # 严格按照你提供的列表顺序排列
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

        # 特征数量校验
        if input_data.shape[1] != model.n_features_in_:
            st.error(f"特征数量不匹配！模型需要 {model.n_features_in_} 个特征，当前输入了 {input_data.shape[1]} 个。")
        else:
            # 执行预测 (获取概率)
            try:
                prob_array = model.predict_proba(input_data)[:, 1]
                
                # 【关键修复】将 numpy 的 float32 转换为 python 原生 float
                prob = float(prob_array[0]) 
                prob_percent = prob * 100

                # 可视化结果
                st.markdown("## 📊 预测结果")
                
                # 分三列展示，中间放大结果
                l_col, m_col, r_col = st.columns([1, 2, 1])
                
                with m_col:
                    # 根据概率高低显示不同颜色
                    if prob_percent >= 70:
                        st.success(f"### 不留置引流管成功概率")
                        st.markdown(f"<h1 style='text-align: center; color: #0f766e;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                        st.info("💡 提示：成功概率较高，建议结合临床情况综合判断。")
                    elif prob_percent >= 40:
                        st.warning(f"### 不留置引流管成功概率")
                        st.markdown(f"<h1 style='text-align: center; color: #d97706;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                        st.info("💡 提示：成功概率中等，需谨慎评估。")
                    else:
                        st.error(f"### 不留置引流管成功概率")
                        st.markdown(f"<h1 style='text-align: center; color: #dc2626;'>{prob_percent:.1f}%</h1>", unsafe_allow_html=True)
                        st.info("💡 提示：成功概率较低，建议做好留置引流管准备。")

                    # 【关键修复】这里传入原生 float 而不是 numpy float32
                    st.progress(prob)

            except Exception as e:
                st.error(f"预测过程中发生错误：{str(e)}")