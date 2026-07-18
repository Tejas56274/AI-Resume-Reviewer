import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io

# --- Configuration ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Helper Functions ---
def clean_json_response(text):
    if not text or not text.strip():
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    return text

def get_working_model():
    """Finds first available model on your account."""
    preferred = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    try:
        available = [m.name.replace("models/", "") for m in genai.list_models()
                     if "generateContent" in m.supported_generation_methods]
        st.sidebar.markdown("**Available Models:**")
        for a in available:
            st.sidebar.code(a)
        for p in preferred:
            if p in available:
                return p
        if available:
            return available[0]
    except Exception as e:
        st.error(f"Could not fetch models: {e}")
    return "gemini-1.5-flash-latest"

def get_ai_response(input_text, pdf_content, prompt):
    full_prompt = f"Job Context: {input_text}\n\nResume:\n{pdf_content}\n\nInstructions:\n{prompt}"
    model_name = get_working_model()
    st.info(f"Using model: **{model_name}**")
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(full_prompt)
        if not response.candidates:
            return "ERROR: No response received. API key may be invalid."
        candidate = response.candidates[0]
        finish_reason = str(candidate.finish_reason)
        if finish_reason not in ("FinishReason.STOP", "1", "STOP"):
            return f"ERROR: Response blocked. Reason: {finish_reason}"
        if not candidate.content or not candidate.content.parts:
            return f"ERROR: Empty response. Finish reason: {finish_reason}"
        return candidate.content.parts[0].text
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def get_optimized_resume_text(resume_text, jd_context, missing_keywords, suggestions):
    """Ask Gemini to rewrite resume with optimized content."""
    prompt = f"""
You are an expert resume writer. Rewrite the following resume to:
1. Naturally incorporate these missing keywords: {', '.join(missing_keywords)}
2. Apply these suggestions: {'; '.join(suggestions)}
3. Keep the same structure, format, and person's actual experience — do NOT fabricate anything
4. Return ONLY the optimized resume text, no explanations

Job Description Context: {jd_context}

Original Resume:
{resume_text}
"""
    model_name = get_working_model()
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        return resume_text
    except Exception:
        return resume_text

def generate_optimized_pdf(optimized_text, original_filename):
    """Generate a clean PDF from optimized resume text."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    DARK = colors.HexColor("#111111")
    BLUE = colors.HexColor("#1a56db")
    GRAY = colors.HexColor("#444444")

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=10, textColor=GRAY, leading=14, spaceAfter=4)
        base.update(kw)
        return ParagraphStyle(name, **base)

    body_s = S("body", alignment=TA_JUSTIFY)
    heading_s = S("heading", fontName="Helvetica-Bold", fontSize=13, textColor=DARK,
                  spaceBefore=10, spaceAfter=4)
    subheading_s = S("subheading", fontName="Helvetica-Bold", fontSize=10.5,
                     textColor=BLUE, spaceBefore=6, spaceAfter=2)
    bullet_s = S("bullet", leftIndent=14, firstLineIndent=-14, spaceAfter=2)

    story = []
    lines = optimized_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        # Detect section headings (ALL CAPS or ends with :)
        if line.isupper() and len(line) > 3:
            story.append(Paragraph(line, heading_s))
            story.append(HRFlowable(width="100%", thickness=0.8, color=BLUE,
                                    spaceBefore=1, spaceAfter=4))
        elif line.endswith(':') and len(line) < 40:
            story.append(Paragraph(line, subheading_s))
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            clean = line.lstrip('•-* ').strip()
            story.append(Paragraph(f"• {clean}", bullet_s))
        else:
            safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_line, body_s))

    doc.build(story)
    buffer.seek(0)
    return buffer

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# --- Prompts ---
input_prompt = """
You are an expert HR Manager with 20 years of experience in tech recruitment.
Analyze the provided resume carefully against the job context given.
Return ONLY a valid JSON object with exactly these keys and value types:
{
  "ATS Score": "85%",
  "Strengths": ["point 1", "point 2", "point 3"],
  "Weaknesses": ["point 1", "point 2"],
  "Missing Keywords": ["keyword1", "keyword2", "keyword3"],
  "Suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}
Rules:
- ATS Score must be a percentage string e.g. "78%"
- All other values must be arrays of strings
- Do NOT include markdown, explanations, or any text outside the JSON object
- Do NOT wrap in code fences
"""

# --- UI Setup ---
st.set_page_config(page_title="AI Resume Reviewer", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        font-size: 1.2em;
    }
    [data-testid="stMetricValue"] {
        font-size: 50px;
        color: #00ff00;
    }
    .download-btn > button {
        background-color: #1a56db !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🚀 AI Resume Reviewer")
st.subheader("ATS Scoring + Optimized Resume Download powered by Google Gemini")
st.write("Boost your resume impact and get placement ready!")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Job Settings")
    jd_context = st.text_area(
        "Target Job Role / Description",
        value="Software Engineer with skills in Python, Data Structures, REST APIs, and problem solving.",
        height=150
    )
    st.info("💡 The more detailed your job description, the more accurate your score will be.")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload your resume PDF")
    st.markdown("2. Enter the job description")
    st.markdown("3. Click Analyze button")
    st.markdown("4. Get ATS score + download optimized resume")

# --- File Upload ---
st.markdown("### 📄 Upload Your Resume")
uploaded_file = st.file_uploader(
    "Upload in PDF format",
    type=["pdf"],
    help="Only PDF files are supported. Make sure it is not a scanned image PDF."
)

if uploaded_file:
    st.success(f"✅ File received: **{uploaded_file.name}**")

st.markdown("---")

# --- Analyze Button ---
submit = st.button("🔍 Analyze My Resume", use_container_width=True)

# --- Main Logic ---
if submit:
    if uploaded_file is None:
        st.error("❌ Please upload your resume PDF first!")
    else:
        with st.spinner("🤖 AI is scanning your resume... please wait!"):

            response_text = ""

            try:
                # Step 1: Extract PDF text
                resume_text = input_pdf_text(uploaded_file)

                if not resume_text.strip():
                    st.error("❌ Could not extract text from the PDF.")
                    st.warning("Possible reasons:")
                    st.markdown("- The PDF is scanned or image-based")
                    st.markdown("- The PDF is password protected")
                    st.markdown("- The PDF file is corrupted")
                    st.stop()

                st.info(f"📃 Successfully extracted **{len(resume_text)} characters** from your resume.")

                # Step 2: Get AI response
                response_text = get_ai_response(jd_context, resume_text, input_prompt)

                # Step 3: Debug expander
                with st.expander("🛠️ Raw API Response (Debug)", expanded=False):
                    st.code(response_text, language="json")

                # Step 4: Check for errors
                if response_text.startswith("ERROR:") or response_text.startswith("EXCEPTION:"):
                    st.error(f"🚨 API Error: {response_text}")
                    st.markdown("**Possible fixes:**")
                    st.markdown("- Make sure you used a fresh Gmail account")
                    st.markdown("- Check your API key at [aistudio.google.com](https://aistudio.google.com)")
                    st.markdown("- Try again after a few minutes")
                    st.stop()

                # Step 5: Clean and parse JSON
                cleaned = clean_json_response(response_text)

                if not cleaned:
                    st.error("❌ Response was empty after cleaning.")
                    st.stop()

                res_json = json.loads(cleaned)

                # --- Display Results ---
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # ATS Score
                score = res_json.get("ATS Score", "N/A")
                st.metric(label="🎯 ATS Compatibility Score", value=score)

                # Score color indicator
                try:
                    score_val = int(score.replace("%", "").strip())
                    if score_val >= 80:
                        st.success("🟢 Excellent! Your resume is well-optimized for ATS systems.")
                    elif score_val >= 60:
                        st.warning("🟡 Average. A few improvements can significantly boost your score.")
                    else:
                        st.error("🔴 Low score. Your resume needs significant improvements.")
                except:
                    pass

                st.markdown("---")

                # Strengths & Weaknesses
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("✅ Key Strengths")
                    strengths = res_json.get("Strengths", [])
                    if strengths:
                        for s in strengths:
                            st.success(f"• {s}")
                    else:
                        st.info("No strengths identified.")

                with col2:
                    st.subheader("⚠️ Areas for Improvement")
                    weaknesses = res_json.get("Weaknesses", [])
                    if weaknesses:
                        for w in weaknesses:
                            st.warning(f"• {w}")
                    else:
                        st.info("No major weaknesses found!")

                st.markdown("---")

                # Missing Keywords
                st.subheader("🔍 Missing Critical Keywords")
                keywords = res_json.get("Missing Keywords", [])
                if keywords:
                    keyword_html = " ".join([
                        f'<span style="background-color:#ff4b4b; color:white; padding:4px 10px; '
                        f'border-radius:12px; margin:4px; display:inline-block;">{k}</span>'
                        for k in keywords
                    ])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 Great! Your resume is already keyword-rich.")

                st.markdown("---")

                # Suggestions
                st.subheader("💡 Expert Suggestions")
                suggestions = res_json.get("Suggestions", [])
                if suggestions:
                    for i, sug in enumerate(suggestions, 1):
                        st.info(f"**{i}.** {sug}")
                else:
                    st.info("No additional suggestions.")

                st.markdown("---")

                # ✨ NEW FEATURE — Optimized Resume Download
                st.markdown("## ✨ Download Optimized Resume")
                st.markdown("AI will rewrite your resume with missing keywords and suggestions applied — **your background and formatting stays the same.**")

                with st.spinner("✍️ Generating your optimized resume..."):
                    try:
                        optimized_text = get_optimized_resume_text(
                            resume_text,
                            jd_context,
                            keywords if keywords else [],
                            suggestions if suggestions else []
                        )

                        pdf_buffer = generate_optimized_pdf(
                            optimized_text,
                            uploaded_file.name
                        )

                        original_name = uploaded_file.name.replace(".pdf", "")
                        download_filename = f"{original_name}_optimized.pdf"

                        st.download_button(
                            label="📥 Download Optimized Resume (PDF)",
                            data=pdf_buffer,
                            file_name=download_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )

                        st.success("✅ Optimized resume ready! Click above to download.")
                        st.caption("💡 The optimized resume has the same content — keywords and suggestions are naturally woven in.")

                    except Exception as e:
                        st.error(f"❌ Could not generate optimized resume: {str(e)}")
                        st.info("Your ATS analysis above is still complete and accurate.")

                st.markdown("---")
                st.caption("✨ Powered by Google Gemini | Built with Streamlit")

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON Parse Error: {e}")
                st.warning("This was the raw response received:")
                st.code(response_text)

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
