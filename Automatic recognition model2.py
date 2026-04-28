import streamlit as st
import joblib
import numpy as np
import os
import openai
import json
import base64
import shap
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ==========================================
# API配置
# ==========================================
API_KEY = "sk-epfqcfmgrmixsgyfvvkldzmintyrxrnoqbucvkejwvrmsmxs" 
BASE_URL = "https://api.siliconflow.cn/v1" 
MODEL = "Qwen/Qwen3.5-27B"


client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
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

KEY_MAPPING = {
    "年龄": "Age",
    "性别": "Gender",
    "体重指数": "BMI",
    "BMI": "BMI",
    "FEV1实测值": "FEV1_residual",
    "FEV1实测": "FEV1_residual",
    "FEV1_residual": "FEV1_residual",
    "FEV1预计值": "FEV1_predicted",
    "FEV1预计": "FEV1_predicted",
    "FEV1_predicted": "FEV1_predicted",
    "FEV1/FVC": "FEV1_FVC",
    "FEV1_FVC": "FEV1_FVC",
    "淋巴结采样": "Lymph_node_sampling",
    "Lymph_node_sampling": "Lymph_node_sampling",
    "肿瘤大小": "Tumor_size",
    "Tumor_size": "Tumor_size",
    "手术时间": "Operation_time",
    "Operation_time": "Operation_time",
    "术中失血量": "Blood_loss",
    "失血量": "Blood_loss",
    "Blood_loss": "Blood_loss",
    "止血材料纤丝类": "Hemo_filament",
    "止血材料-纤丝类": "Hemo_filament",
    "Hemo_filament": "Hemo_filament",
    "止血材料膜类": "Hemo_membrane",
    "止血材料-膜类": "Hemo_membrane",
    "Hemo_membrane": "Hemo_membrane",
    "白细胞": "WBC",
    "WBC": "WBC",
    "血红蛋白": "Hb",
    "Hb": "Hb",
    "血小板": "PLT",
    "PLT": "PLT",
    "谷丙转氨酶": "ALT",
    "ALT": "ALT",
    "谷草转氨酶": "AST",
    "AST": "AST",
    "总蛋白": "Total_protein",
    "Total_protein": "Total_protein",
    "白蛋白": "Albumin",
    "Albumin": "Albumin",
    "钾": "K",
    "K": "K",
    "钙": "Ca",
    "Ca": "Ca",
    "血糖": "Glucose",
    "Glucose": "Glucose"
}

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
你是医学数据提取助手。请从图片中提取以下字段，并严格按照指定键名返回 JSON。

【字段映射表】（图片中的中文 → 输出键名）
- 年龄 → Age
- 性别 → Gender（男=0，女=1）
- 体重指数/BMI → BMI
- FEV1实测值 → FEV1_residual（单位：L）
- FEV1预计值 → FEV1_predicted（单位：%）
- FEV1/FVC → FEV1_FVC
- 淋巴结采样 → Lymph_node_sampling（是=1，否=0）
- 肿瘤大小 → Tumor_size（单位：mm，如为cm请×10）
- 手术时间 → Operation_time（单位：min）
- 术中失血量 → Blood_loss（单位：ml）
- 止血材料纤丝类 → Hemo_filament（使用=1，未使用=0）
- 止血材料膜类 → Hemo_membrane（使用=1，未使用=0）
- 白细胞 → WBC
- 血红蛋白 → Hb
- 血小板 → PLT
- 谷丙转氨酶 → ALT
- 谷草转氨酶 → AST
- 总蛋白 → Total_protein
- 白蛋白 → Albumin
- 钾 → K
- 钙 → Ca
- 血糖 → Glucose

【要求】
1. 只返回纯 JSON，不要包含任何解释或 markdown 标记
2. 仅返回图片中明确存在的字段
3. 数值类型保持为数字，不要加单位
4. 禁止编造数据
5. 示例格式：{"Age": 65, "WBC": 6.5, "Hb": 135}
"""

                    response = client.chat.completions.create(
                        model=MODEL,
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

                    if result.startswith("```"):
                        result = result.split("\n", 1)[1]
                    if result.endswith("```"):
                        result = result.rsplit("\n", 1)[0]
                    
                    if "<|begin_of_box|>" in result:
                        result = result.split("<|begin_of_box|>")[-1]
                    if "<|end_of_box|>" in result:
                        result = result.split("<|end_of_box|>")[0]
                    
                    result = result.strip()

                    try:

                        data = json.loads(result)

                        for k,v in data.items():

                            mapped_key = KEY_MAPPING.get(k, k)

                            if mapped_key not in extracted_all:
                                extracted_all[mapped_key] = v

                    except json.JSONDecodeError as e:
                        st.error(f"JSON 解析失败：{e}")
                        with st.expander("查看 AI 原始输出"):
                            st.code(result)

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

                    with st.expander("🔧 调试：查看 AI 原始输出"):
                        st.code(result)

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
# SHAP Explainer加载（已预训练保存）
# ==========================================
@st.cache_resource
def load_shap_explainer():
    """
    直接加载预训练的SHAP Explainer
    避免每次启动都重新计算，大幅提升加载速度
    """
    EXPLAINER_PATH = os.path.join(current_dir, 'shap_explainer.joblib')
    
    if not os.path.exists(EXPLAINER_PATH):
        st.error(f"❌ 找不到SHAP Explainer文件：{EXPLAINER_PATH}")
        st.info("""
        **请先在训练脚本中保存explainer：**
        ```python
        import joblib
        explainer = shap.LinearExplainer(final_model, X_scaled)
        joblib.dump(explainer, 'shap_explainer.joblib')
        ```
        """)
        return None, None
    
    # 加载explainer
    explainer = joblib.load(EXPLAINER_PATH)
    
    # 特征名称（与训练时一致）
    feature_names = [
        'Age', 'Gender', 'BMI', 
        'Preoperative FEV1 residual', 'Preoperative FEV1 predicted (%)', 'Preoperative FEV1/FVC', 
        'Lymph node sampling', 'Tumor size(mm)', 
        'Operation time(min)', 'Intraoperative blood loss(ml)',
        'Hemostatic material usage_ Filament type', 'Hemostatic materials usage_ Membrane type',
        'Preoperative white blood cell count', 'Preoperative hemoglobin',
        'Preoperative platelets', 'Preoperative alanine aminotransferase', 'Preoperative Aspartate Aminotransferase',
        'Preoperative total protein', 'Preoperative albumin',
        'Preoperative potassium', 'Preoperative calcium', 'Preoperative glucose'
    ]
    
    # 如果有独热编码后的特征，需要更新feature_names
    # 这里使用简化版本，实际应该从explainer中获取
    if hasattr(explainer, 'model') and hasattr(explainer.model, 'coef_'):
        n_features = explainer.model.coef_.shape[1]
        if n_features != len(feature_names):
            st.warning(f"⚠️ 特征数量不匹配：期望{len(feature_names)}，实际{n_features}")
            feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    return explainer, feature_names

explainer, feature_names = load_shap_explainer()

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
            format_func=lambda x:"女 (Female)" if x==1 else "男 (Male)",
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
    
    # ==========================================
    # SHAP决策图生成
    # ==========================================
    st.markdown("---")
    st.markdown("## 🔍 SHAP决策分析 | SHAP Decision Analysis")
    
    if explainer is None:
        st.warning("⚠️ SHAP Explainer未加载，无法生成决策图")
    else:
        with st.spinner("正在生成SHAP决策图..."):
            try:
                # 计算SHAP值
                shap_values = explainer.shap_values(final)
                
                # 处理二分类模型的SHAP值（取正类）
                if isinstance(shap_values, list):
                    shap_values_sample = shap_values[1]
                else:
                    shap_values_sample = shap_values
                
                # 获取基线值
                base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]
                
                # 生成SHAP决策图
                fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
                shap.decision_plot(
                    base_value=base_value,
                    shap_values=shap_values_sample[0],
                    feature_names=feature_names,
                    show=False
                )
                plt.title('SHAP Decision Plot', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                # 解读说明
                st.markdown("---")
                st.markdown("### 💡 如何解读SHAP决策图？")
                
                st.markdown("""
                **决策图 (Decision Plot)**：
                - **基线（灰色竖线，x≈7.5）**：代表模型在无任何特征输入时的基准预测值（即所有样本的平均预测输出），是衡量特征影响的参照零点。
                - **粉色折线**：从基线出发，沿纵轴（特征列表）自下而上，依次累加每个特征的 SHAP 值，最终折线终点的 x 坐标，就是该样本的最终模型预测值。
                
                **SHAP 值的正负含义**：
                - **折线向右偏移（x 值增大）**：该特征对预测结果产生正向推动作用，提升了模型输出；
                - **折线向左偏移（x 值减小）**：该特征对预测结果产生负向抑制作用，降低了模型输出；
                - **折线几乎垂直（x 无明显变化）**：该特征对该样本的预测几乎无影响。
                
                **特征贡献排序**：纵轴按特征对该样本预测的影响程度从大到小排列（顶部特征贡献最大，底部特征贡献最小）。
                
                **顶部色条**：辅助理解特征取值与 SHAP 值的关联，蓝色对应低 SHAP 值（负向影响），红色对应高 SHAP 值（正向影响）。
                
                **临床意义**：
                - 了解哪些因素对该患者的预测结果影响最大
                - 帮助医生理解模型决策依据
                - 识别可干预的风险因素
                """)

                # 特征贡献度表格（不显示特征值）
                st.markdown("### 📋 特征贡献度详情 | Feature Contribution Details")
                
                feature_importance = pd.DataFrame({
                    '特征 | Feature': feature_names,
                    'SHAP值 | SHAP Value': shap_values_sample[0],
                    '影响方向 | Impact': ['↑ 增加概率' if sv > 0 else '↓ 降低概率' for sv in shap_values_sample[0]],
                    '影响程度 | Impact Strength': np.abs(shap_values_sample[0])
                })
                
                # 按SHAP值绝对值排序
                feature_importance = feature_importance.sort_values('影响程度 | Impact Strength', ascending=False)
                
                # 移除影响程度列（仅用于排序）
                feature_importance = feature_importance.drop('影响程度 | Impact Strength', axis=1)
                
                # 显示前15个最重要特征
                st.dataframe(
                    feature_importance.head(15),
                    use_container_width=True,
                    hide_index=True
                )
                
                # ==========================================
                # AI智能解读
                # ==========================================
                st.markdown("---")
                st.markdown("### 🤖 AI智能解读 | AI Clinical Interpretation")
                
                with st.spinner("AI正在生成临床解读..."):
                    try:
                        # 构建特征值字典（原始值，非标准化）
                        input_values = {k: st.session_state[k] for k in PARAMS}
                        
                        # 构建特征贡献度列表（前10个最重要特征）
                        top_features = feature_importance.head(10).to_dict('records')
                        
                        # 构建prompt
                        system_prompt = f"""你是胸外科临床决策辅助AI助手。请根据以下信息，用简洁专业的中文生成一段临床解读。

【预测任务说明】
- 模型预测的是"术后不留置引流管的成功概率"
- 结局=1：患者情况可以术后不留置引流管
- 结局=0：患者术后需要留置引流管

【特征解释】
- Hemostatic materials usage_ Membrane type_1：术中膜类材料使用。1=使用，0=未使用
- Hemostatic material usage_ Filament type_1：术中纤丝类材料使用。1=使用，0=未使用
- Preoperative alanine aminotransferase：术前丙氨酸氨基转移酶。正常范围9-50
- Preoperative Aspartate Aminotransferase：术前天冬氨酸氨基转移酶。正常范围15-40
- Age：患者手术时年龄
- Preoperative albumin：术前血白蛋白水平。正常范围40-55
- Intraoperative blood loss(ml)：术中失血量（毫升），越少越好
- BMI：身高体重指数
- Preoperative calcium：术前血钙水平。正常范围2.2-2.7
- Preoperative FEV1 residual：术前用力呼气一秒量绝对值（L）
- Preoperative FEV1 predicted (%)：术前用力呼气一秒量占预计值百分比
- Preoperative FEV1/FVC：术前用力呼气一秒率
- Gender_2：患者性别。0=男性，1=女性
- Preoperative glucose：术前空腹血糖水平。正常范围3.9-6.1
- Preoperative hemoglobin：术前血红蛋白水平。正常范围：男130-175，女115-150
- Preoperative potassium：术前血钾水平。正常范围3.5-5.3
- Lymph node sampling_2：术中淋巴结采样。1=采样，0=未采样
- Operation time(min)：手术时间（分钟）
- Preoperative platelets：术前血小板水平。正常范围125-350
- Preoperative total protein：术前血总蛋白水平。正常范围65-85
- Tumor size(mm)：肿瘤大小（mm）。模型适用0-20mm，超出范围结果仅供参考
- Preoperative white blood cell count：术前血白细胞水平。正常范围3.5-9.5

【输出要求】
1. 先总结患者基本情况和预测结果
2. 分析影响预测结果的关键因素（结合SHAP值和临床意义）
3. 指出异常指标（如有）
4. 给出临床建议
5. 语言简洁专业，200-300字
6. 不要使用markdown格式，纯文本输出"""
                        
                        user_prompt = f"""【患者输入特征值】
{json.dumps(input_values, ensure_ascii=False, indent=2)}

【预测结果】
不留置引流管成功概率：{prob*100:.1f}%

【SHAP特征贡献度（前10个最重要特征）】
{json.dumps(top_features, ensure_ascii=False, indent=2)}

请生成临床解读："""
                        
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3
                        )
                        
                        interpretation = response.choices[0].message.content.strip()
                        
                        # 显示解读结果
                        st.markdown(f"""
                        <div style="
                            background-color: #f0f7ff;
                            border-left: 4px solid #3498db;
                            padding: 16px;
                            border-radius: 4px;
                            line-height: 1.8;
                        ">
                        {interpretation}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"AI解读生成失败：{str(e)}")
                
                               
            except Exception as e:
                st.error(f"SHAP分析生成失败：{str(e)}")
                st.info("请确保已安装shap库：pip install shap")