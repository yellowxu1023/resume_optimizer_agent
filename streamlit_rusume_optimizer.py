import streamlit as st
import os
import json
import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from docxtpl import DocxTemplate


# ==========================================
# --- 新增：总体总结与建议模型 ---
class OptimizationReport(BaseModel):
    summary: str = Field(description="总体优化总结")
    gap_analysis: List[str] = Field(description="原简历与JD的差异/缺失分析")
    actionable_advice: List[str] = Field(description="给候选人的可执行建议")


# 1. 定义数据结构 (Pydantic Models) - 保持不变
# ==========================================
class PersonalInfo(BaseModel):
    name: str
    phone: str
    email: str
    education_summary: str
    experience_years: str
    age: str


class WorkExperience(BaseModel):
    company: str
    position: str
    date_range: str
    responsibilities: List[str]


class ProjectExperience(BaseModel):
    name: str = Field(description="项目名称，如：某OEM镀膜及先进材料产品结构与工艺改进项目")
    role: str = Field(description="项目角色，如：项目统筹与技术负责人")
    date_range: str = Field(description="项目时间")
    description: str = Field(description="项目内容描述")
    contributions: List[str] = Field(description="核心贡献列表，使用STAR法则精简")


class EducationExperience(BaseModel):
    school: str
    major: str
    degree: str
    date_range: str


class ResumeData(BaseModel):
    personal_info: PersonalInfo
    core_advantages: List[str]
    professional_skills: List[str]
    work_experiences: List[WorkExperience]
    project_experiences: List[ProjectExperience]  # 新增：项目经历
    education_experiences: List[EducationExperience]  # 新增：教育经历
    certifications_and_languages: List[str]  # 新增：证书与语言


# ==========================================
# 2. Agent 核心类 (针对 DeepSeek 优化)
# ==========================================
class ResumeOptimizationAgent:
    def __init__(self, api_key: str, model_name: str = "deepseek-v4-pro"):
        """
        初始化 Agent
        :param api_key: DeepSeek API Key
        :param model_name: 默认为 deepseek-chat，若 DeepSeek v4 有专属模型名请直接替换
        """
        if not api_key:
            raise ValueError("API Key 不能为空！")

        # 【修改点 1】：配置 DeepSeek 的 Base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name
        print(f"✅ Agent 初始化成功，使用 DeepSeek 模型: {self.model_name}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从 PDF 提取文本"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到 PDF 文件: {pdf_path}")

        print(f"📄 正在解析 PDF: {pdf_path} ...")
        text_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            return "\n".join(text_content)
        except Exception as e:
            raise RuntimeError(f"PDF 解析失败: {e}")

    def optimize_resume_step(self, raw_text: str, jd_text: str) -> ResumeData:
        """调用 DeepSeek 进行结构化提取和定向优化（带智能纠错解析）"""
        print("🧠 正在呼叫 DeepSeek 进行信息提取与 JD 对齐优化 ...")

        schema_json = json.dumps(ResumeData.model_json_schema(), ensure_ascii=False)

        # 【升级点1：在 Prompt 中明确禁止加外壳】
        prompt = f"""
        你是一个极度严谨的简历优化专家。
        任务：将[简历原文]的内容，针对[目标岗位JD]进行结构化提取和润色。

        【核心底线：严禁编造】
        1. 数据真实性：绝对禁止捏造任何原简历中不存在的数据（如百分比、金额、人数、年限）。如果原简历中没有量化指标，你可以通过强化“动词”和“成果描述”来体现专业度，但严禁自行填充数字。
        2. 经历真实性：严禁增加候选人没有做过的项目或没待过的公司。
        3. 技能真实性：仅提取原简历中提到的技能。若 JD 要求某个技能而原简历中没有，请不要在简历中添加。

        【润色策略：语意映射与无痕STAR】
        - 术语对齐：将原简历中非标准的描述转换为 JD 中的专业术语。
        - 优先级重排：将原简历中与 JD 匹配度最高的经历和技能放在最显眼的位置。
        - 动作强化：使用更具领导力和执行力的动词（如：主导、驱动、优化、重构、整合）。
        - 隐含 STAR 法则：工作经历在逻辑上需符合“情境-任务-行动-结果”的闭环，但【绝对禁止】在文本中直接输出“STAR”、“情境”、“任务”、“行动”、“结果”、“背景”等字眼！请将其融合成一句自然、连贯、精炼的专业描述（例如直接输出：“针对系统卡顿问题，引入重构机制，将加载速度提升30%”）。

        【输出格式要求（极其重要）】
        你必须严格按照以下 JSON Schema 格式输出结果。
        注意：请直接在根节点输出 `personal_info` 等字段，**绝对禁止**将结果包裹在 `optimized_resume`、`ResumeData` 或任何额外嵌套的字典外壳中！不要输出任何多余的解释文字！

        {schema_json}

        【简历原文】
        {raw_text}

        【目标岗位JD】
        {jd_text}
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个严格的数据提取引擎，总是以纯 JSON 格式响应。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2  # 调低温度，让格式更稳定
            )

            result_content = completion.choices[0].message.content

            # 【升级点2：智能剥离外壳逻辑】
            # 先将文本解析为基础的 Python 字典
            data_dict = json.loads(result_content)

            # 检查必需的字段是否在第一层。如果不在，说明模型多包了一层。
            if "personal_info" not in data_dict:
                # 遍历寻找包含目标数据的内层字典
                for key, value in data_dict.items():
                    if isinstance(value, dict) and "personal_info" in value:
                        print(f"⚠️ 拦截到大模型多加了外壳 ['{key}']，已自动剥离修复！")
                        data_dict = value
                        break

            print("✨ DeepSeek 处理完成，JSON 结构校验通过！")
            # 使用 Pydantic 的 model_validate (接受字典) 而不是 model_validate_json (接受字符串)
            return ResumeData.model_validate(data_dict)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"大模型返回的不是合法的 JSON: {e}\n原始返回内容:\n{result_content}")
        except Exception as e:
            raise RuntimeError(f"DeepSeek 调用失败或数据结构错误: {e}\n原始返回内容:\n{result_content}")

    def render_to_word(self, optimized_data: ResumeData, template_path: str, output_path: str):
        """将数据渲染至 Word 模板"""
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"找不到 Word 模板文件: {template_path}")

        print(f"📝 正在将数据填入 Word 模板: {template_path} ...")
        try:
            doc = DocxTemplate(template_path)
            context = optimized_data.model_dump()
            doc.render(context)
            doc.save(output_path)
            print(f"🎉 简历优化大功告成！已保存至: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Word 生成失败: {e}")

    # ------------------------------------------
    # 第二步：专门生成差距分析报告
    # ------------------------------------------
    def generate_report_step(self, raw_text: str, jd_text: str, optimized_data: ResumeData) -> OptimizationReport:
        print("🧠 [Step 2] 正在呼叫 DeepSeek 进行 Gap 分析并生成面试建议 ...")

        optimized_json_str = json.dumps(optimized_data.model_dump(), ensure_ascii=False)

        prompt = f"""
        你是一个顶级资深猎头顾问。
        对比候选人的[原简历]、[目标岗位JD]以及系统刚刚生成的[优化后简历]，输出一份专业的《总体总结与建议报告》。

        【分析要求】
        1. 总结：解释本次润色重点强化了哪些维度的表达。
        2. 差距分析 (Gap)：找准原简历中缺失的、但 JD 中强要求的技能或经验。
        3. 行动建议：告诉候选人需要手动补充哪些具体的量化数据，或面试时重点准备哪些话术规避劣势。

        【输出格式要求】
        严格输出如下格式的纯 JSON：
        {{
            "summary": "总体优化思路的总结...",
            "gap_analysis": ["差距1", "差距2"],
            "actionable_advice": ["面试建议1", "补充数据建议2"]
        }}

        【目标岗位JD】
        {jd_text}

        【原简历】
        {raw_text}

        【优化后简历】
        {optimized_json_str}
        """

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        report_dict = json.loads(completion.choices[0].message.content)

        actual_report = report_dict
        if "summary" not in report_dict:
            for v in report_dict.values():
                if isinstance(v, dict) and "summary" in v:
                    actual_report = v
                    break

        print("✨ [Step 2] 咨询报告生成完成！")
        return OptimizationReport.model_validate(actual_report)

    def run(self, pdf_path: str, jd_text: str, template_path: str, output_path: str):
        try:
            print("=" * 50)
            print("🚀 简历优化双引擎工作流启动")
            print("=" * 50)

            raw_text = self.extract_text_from_pdf(pdf_path)

            # Step 1: 优化生成
            optimized_data = self.optimize_resume_step(raw_text, jd_text)
            self.render_to_word(optimized_data, template_path, output_path)

            # Step 2: 报告生成
            report = self.generate_report_step(raw_text, jd_text, optimized_data)

            print("\n" + "🌟" * 25)
            print(f"【AI 猎头顾问 - 简历优化洞察报告】")
            print("🌟" * 25 + "\n")

            print("💡 总体总结：")
            print(f"  {report.summary}\n")

            print("⚠️ 差距分析 (Gap Analysis)：")
            for i, gap in enumerate(report.gap_analysis, 1):
                print(f"  {i}. {gap}")
            print()

            print("🎯 行动建议 (Actionable Advice)：")
            for i, advice in enumerate(report.actionable_advice, 1):
                print(f"  {i}. {advice}")

            print("\n" + "=" * 50)
            print("✅ 任务圆满结束")
            print("=" * 50)

        except Exception as e:
            print(f"\n❌ 任务异常终止: {e}")


# ==========================================
# 2. Streamlit 网页 UI 设计区
# ==========================================
st.set_page_config(page_title="AI 顶级猎头 - 简历优化引擎", page_icon="💼", layout="centered")

st.title("💼 AI 顶级猎头 - 简历优化引擎")
st.markdown("上传你的PDF简历，输入目标岗位JD，AI将为你重构一份直击HR痛点的满分简历，并出具深度诊断报告！")


try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    st.error("未检测到 DEEPSEEK_API_KEY 配置，请检查云端Secrets配置")
    st.stop()
# 主界面：输入区
st.subheader("1. 资料上传")
uploaded_file = st.file_uploader("上传原简历 (仅支持 PDF 格式)", type=["pdf"])
jd_text = st.text_area("粘贴目标岗位的 JD (职位描述)", height=200, placeholder="请粘贴完整的岗位要求和职责描述...")

# ==========================================
# 动作按钮
# ==========================================
if st.button("🚀 开始一键优化", type="primary"):
    if not uploaded_file:
        st.warning("📄 请上传你的原版 PDF 简历！")
    elif not jd_text.strip():
        st.warning("📝 请输入目标岗位 JD！")
    else:
        with st.spinner("🧠 AI 正在深度思考与重构简历，请稍候（约需 1-2 分钟）..."):
            try:
                temp_pdf_path = "temp_uploaded_resume.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                agent = ResumeOptimizationAgent(api_key=API_KEY)

                raw_text = agent.extract_text_from_pdf(temp_pdf_path)
                optimized_data = agent.optimize_resume_step(raw_text, jd_text)

                output_docx_path = "优化后满分简历.docx"
                template_path = "template_full_style.docx"
                agent.render_to_word(optimized_data, template_path, output_docx_path)

                report = agent.generate_report_step(raw_text, jd_text, optimized_data)

                os.remove(temp_pdf_path)

                # 🌟 核心改动 1：把生成好的数据存入 Streamlit 的“记忆胶囊”中
                st.session_state['generated_report'] = report
                with open(output_docx_path, "rb") as file:
                    st.session_state['generated_docx'] = file.read()

                st.success("🎉 简历优化大功告成！")

            except Exception as e:
                st.error(f"❌ 运行中出现错误，请检查 API Key 或重试：\n{e}")

# ==========================================
# 3. 结果展示区 (🌟 核心改动 2：移到了按钮判断的外面)
# 只要记忆胶囊里有数据，不管怎么点击刷新，报告都会死死钉在页面上！
# ==========================================
if 'generated_report' in st.session_state and 'generated_docx' in st.session_state:
    # 提供下载按钮 (直接从记忆胶囊中读取 Word 二进制数据)
    st.download_button(
        label="📥 下载精美 Word 简历",
        data=st.session_state['generated_docx'],
        file_name="优化后满分简历.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    # 展示诊断报告
    st.divider()
    st.subheader("📊 AI 猎头洞察报告")

    # 从记忆胶囊中取出报告对象
    saved_report = st.session_state['generated_report']

    st.info(f"**💡 总体总结：**\n\n{saved_report.summary}")
    st.warning("**⚠ 差距分析 (Gap Analysis)：**\n\n" + "\n".join([f"- {g}" for g in saved_report.gap_analysis]))
    st.success("**🎯 行动建议 (Actionable Advice)：**\n\n" + "\n".join([f"- {a}" for a in saved_report.actionable_advice]))
