import streamlit as st
import joblib
import numpy as np
import os
import openai
import json
import base64

# ==========================================
# 豆包API配置
# ==========================================
DOUBAO_API_KEY = "7bae6018-2ccc-44e5-9fc0-4d77bf3720b6" 
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3" 
DOUBAO_MODEL = "ep-20260308165721-tz8ds" 

client = openai.OpenAI(
    api_key=DOUBAO_API_KEY,
    base_url=DOUBAO_BASE_URL
)

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="机器学习肺楔形切除术引流管预测 | ML Prediction for Lung Wedge Resection Drainage",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 所有变量
# ==========================================
PARAMS = [
'Age','Gender','BMI',
'FEV1_residual','FEV1_predicted','FEV1_FVC',
'Lymph_node_sampling','Tumor_size','Operation_time',
'Blood_loss','Hemo_filament','Hemo_membrane',
'WBC','Hb','PLT','ALT','AST',
'Total_protein','Albumin','K','Ca','Glucose'
]

# ==========================================
# Session State 初始化
# ==========================================
for p in PARAMS:
    if p not in st.session_state:
        st.session_state[p] = None

if "ai_filled" not in st.session_state:
    st.session_state.ai_filled = set()

# ==========================================
# Logo
# ==========================================
current_dir = os.path.dirname(__file__)
logo_path = os.path.join(current_dir, "logo.png")

if os.path.exists(logo_path):

    with open(logo_path,"rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;">
        <img src="data:image/png;base64,{img_base64}" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ==========================================
# 标题
# ==========================================
st.title("🫁 机器学习肺楔形切除术不留置引流管成功概率预测 | ML Prediction of Successful Drain-Free after Lung Wedge Resection")

st.markdown("---")

# ==========================================
# AI图片识别模块
# ==========================================
st.markdown("## 📸 智能参数录入 | AI Auto-Fill (Beta)")

col_upload,col_tips = st.columns([1,1])

with col_upload:

    uploaded_files = st.file_uploader(
        "上传检验报告单/病历/手术记录图片（支持多张）",
        type=['png','jpg','jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write("已上传图片")

        for img in uploaded_files:
            st.image(img,width=250)

        if st.button("🔍 开始智能识别并填充",type="primary"):

            with st.spinner("AI正在分析图片..."):

                extracted_all = {}

                for file in uploaded_files:

                    base64_image = base64.b64encode(file.read()).decode()

                    system_prompt = """
你是医学数据提取助手。

允许输出的变量：

Age
Gender
BMI
FEV1_residual
FEV1_predicted
FEV1_FVC
Lymph_node_sampling
Tumor_size
Operation_time
Blood_loss
Hemo_filament
Hemo_membrane
WBC
Hb
PLT
ALT
AST
Total_protein
Albumin
K
Ca
Glucose

要求：

只返回JSON
仅返回图片中存在的字段
禁止编造
"""

                    response = client.chat.completions.create(
                        model=DOUBAO_MODEL,
                        messages=[
                            {
                                "role":"user",
                                "content":[
                                    {"type":"text","text":system_prompt},
                                    {
                                        "type":"image_url",
                                        "image_url":{
                                            "url":f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.1
                    )

                    result = response.choices[0].message.content.strip()

                    try:

                        data = json.loads(result)

                        for k,v in data.items():

                            if k not in extracted_all:
                                extracted_all[k] = v

                    except:
                        pass

                update_count = 0

                for k,v in extracted_all.items():

                    if k in PARAMS:

                        if isinstance(v,(int,float)):

                            st.session_state[k] = v
                            st.session_state.ai_filled.add(k)

                            update_count += 1

                if update_count>0:

                    st.success(f"AI识别成功：{update_count} 个参数")

                    with st.expander("查看识别数据"):
                        st.json(extracted_all)

                    st.rerun()

                else:

                    st.warning("未识别到有效参数")


with col_tips:

    st.markdown("""
### 使用说明

1 上传检验报告、手术记录或病历图片  
2 可以上传多张  
3 点击识别  
4 AI自动填入表单  
5 医生补充剩余参数  
6 点击预测
""")

st.markdown("---")

# ==========================================
# AI识别字段高亮展示
# ==========================================
if len(st.session_state.ai_filled)>0:

    st.markdown("### 🤖 AI识别并填入的变量")

    highlight_html = ""

    for p in PARAMS:

        if p in st.session_state.ai_filled:

            highlight_html += f"""
            <span style="
            background-color:#ffe066;
            padding:6px;
            margin:4px;
            border-radius:6px;
            display:inline-block;
            font-weight:600;
            ">
            {p}
            </span>
            """

    st.markdown(highlight_html,unsafe_allow_html=True)

# ==========================================
# 清空表单
# ==========================================
if st.button("🧹 清空所有参数"):

    for p in PARAMS:
        st.session_state[p] = None

    st.session_state.ai_filled = set()

    st.rerun()

st.markdown("---")

# ==========================================
# 模型加载
# ==========================================
@st.cache_resource
def load_models():

    MODEL_PATH = os.path.join(current_dir,'logistic_final_model.joblib')
    SCALER_PATH = os.path.join(current_dir,'scaler_final.joblib')

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model,scaler

model,scaler = load_models()

# ==========================================
# 表单（完全保留原始内容）
# ==========================================
with st.form("prediction_form"):

    st.subheader("👤 一、患者基础信息 | 1. Basic Patient Information")

    col1,col2,col3 = st.columns(3)

    with col1:
        Age = st.number_input(
            "年龄 (Age, 岁) | Age (years)",
            min_value=0,max_value=120,step=1,
            key="Age"
        )

    with col2:
        Gender = st.radio(
            "性别 (Gender) | Gender",
            options=[0,1],
            format_func=lambda x:"女 (Female)" if x==0 else "男 (Male)",
            horizontal=True,
            key="Gender"
        )

    with col3:
        BMI = st.number_input(
            "体重指数 (BMI, kg/m²) | Body Mass Index",
            min_value=10.0,max_value=50.0,step=0.1,
            key="BMI"
        )

    st.markdown("---")

    st.subheader("🫁 二、术前肺功能")

    col1,col2,col3 = st.columns(3)

    with col1:
        FEV1_residual = st.number_input(
            "术前 FEV1 实测值 (L)",
            min_value=0.0,step=0.01,
            key="FEV1_residual"
        )

    with col2:
        FEV1_predicted = st.number_input(
            "术前 FEV1 预计值 (%)",
            min_value=0.0,max_value=200.0,step=0.1,
            key="FEV1_predicted"
        )

    with col3:
        FEV1_FVC = st.number_input(
            "术前 FEV1/FVC",
            min_value=0.0,max_value=2.0,step=0.01,
            key="FEV1_FVC"
        )

    st.markdown("---")

    st.subheader("🏥 三、手术相关指标")

    col1,col2 = st.columns(2)

    with col1:

        Lymph_node_sampling = st.radio(
            "淋巴结采样",
            options=[0,1],
            format_func=lambda x:"否" if x==0 else "是",
            horizontal=True,
            key="Lymph_node_sampling"
        )

        Tumor_size = st.number_input(
            "肿瘤大小 (mm)",
            min_value=0.0,step=0.1,
            key="Tumor_size"
        )

        Operation_time = st.number_input(
            "手术时间 (min)",
            min_value=0.0,step=1.0,
            key="Operation_time"
        )

    with col2:

        Blood_loss = st.number_input(
            "术中失血量 (ml)",
            min_value=0.0,step=1.0,
            key="Blood_loss"
        )

        Hemo_filament = st.radio(
            "止血材料-纤丝类",
            options=[0,1],
            format_func=lambda x:"未使用" if x==0 else "使用",
            horizontal=True,
            key="Hemo_filament"
        )

        Hemo_membrane = st.radio(
            "止血材料-膜类",
            options=[0,1],
            format_func=lambda x:"未使用" if x==0 else "使用",
            horizontal=True,
            key="Hemo_membrane"
        )

    st.markdown("---")

    st.subheader("🧪 四、术前实验室检查")

    col1,col2,col3 = st.columns(3)

    with col1:
        WBC = st.number_input("白细胞",step=0.01,key="WBC")
        Hb = st.number_input("血红蛋白",step=0.1,key="Hb")
        PLT = st.number_input("血小板",step=1.0,key="PLT")

    with col2:
        ALT = st.number_input("ALT",step=0.1,key="ALT")
        AST = st.number_input("AST",step=0.1,key="AST")
        Total_protein = st.number_input("总蛋白",step=0.1,key="Total_protein")

    with col3:
        Albumin = st.number_input("白蛋白",step=0.1,key="Albumin")
        K = st.number_input("K",step=0.01,key="K")
        Ca = st.number_input("Ca",step=0.01,key="Ca")
        Glucose = st.number_input("Glucose",step=0.01,key="Glucose")

    submitted = st.form_submit_button("🔍 计算成功概率")

# ==========================================
# 预测
# ==========================================
if submitted:

    missing = [k for k in PARAMS if st.session_state[k] is None]

    if missing:

        st.error("以下参数未填写")

        st.write(missing)

        st.stop()

    data = np.array([[st.session_state[k] for k in PARAMS]])

    continuous_indices = [0,2,3,4,5,7,8,9,12,13,14,15,16,17,18,19,20,21]
    categorical_indices = [1,6,10,11]

    cont = data[:,continuous_indices]
    cat = data[:,categorical_indices]

    cont_scaled = scaler.transform(cont)

    final = np.zeros((1,22))

    final[:,continuous_indices] = cont_scaled
    final[:,categorical_indices] = cat

    prob = model.predict_proba(final)[0][1]

    st.markdown("## 📊 预测结果")

    st.metric("成功概率",f"{prob*100:.1f}%")

    st.progress(float(prob))