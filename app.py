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
    prompt = f"""
You are an expert resume writer. Rewrite the following resume to:
1. Naturally incorporate these missing keywords: {', '.join(missing_keywords)}
2. Apply these suggestions: {'; '.join(suggestions)}
3. Keep EXACT same structure and sections as original
4. Keep all project names, bullet points, certifications, education SAME
5. Do NOT add new projects or fabricate experience
6. Do NOT use markdown bold (**text**) — plain text only
7. Keep live demo links exactly as they are
8. Return ONLY the resume text, nothing else

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

def generate_optimized_pdf(optimized_text):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.35*inch,
        bottomMargin=0.35*inch
    )

    DARK = colors.HexColor("#111111")
    BLUE = colors.HexColor("#1a56db")
    GRAY = colors.HexColor("#444444")

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=8.5,
                    textColor=GRAY, leading=11, spaceAfter=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    name_s    = S("n", fontName="Helvetica-Bold", fontSize=17, textColor=DARK,
                  alignment=TA_CENTER, spaceAfter=2, leading=20)
    contact_s = S("c", fontSize=8, textColor=GRAY,
                  alignment=TA_CENTER, spaceAfter=4, leading=12)
    sec_s     = S("sec", fontName="Helvetica-Bold", fontSize=9, textColor=BLUE,
                  spaceBefore=4, spaceAfter=1)
    proj_s    = S("proj", fontName="Helvetica-Bold", fontSize=8.5, textColor=DARK,
                  spaceBefore=3, spaceAfter=0)
    stack_s   = S("stack", fontName="Helvetica-Oblique", fontSize=8,
                  textColor=BLUE, spaceAfter=1)
    body_s    = S("body", fontSize=8.2, textColor=GRAY,
                  leading=11, spaceAfter=1, alignment=TA_JUSTIFY)
    bul_s     = S("bul", fontSize=8.2, textColor=GRAY, leading=11,
                  leftIndent=10, firstLineIndent=-10, spaceAfter=1)
    skill_s   = S("sk", fontSize=8.2, textColor=GRAY, leading=11, spaceAfter=1)

    SECTIONS = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROJECTS",
        "CERTIFICATIONS", "EDUCATION", "EXPERIENCE", "ACHIEVEMENTS", "SKILLS"
    ]

    def hr():
        return HRFlowable(width="100%", thickness=0.7, color=BLUE,
                          spaceBefore=1, spaceAfter=2)

    def safe(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def b(text):
        return Paragraph(f"&#x2022; {safe(text)}", bul_s)

    story = []
    lines = [l for l in optimized_text.split('\n')]

    name_done = False
    contact_done = False
    current_section = None
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        # Remove markdown bold
        line = re.sub(r'\*\*', '', line)

        if not line:
            i += 1
            continue

        # Name
        if not name_done:
            story.append(Paragraph(safe(line), name_s))
            name_done = True
            i += 1
            continue

        # Contact line
        if not contact_done:
            story.append(Paragraph(safe(line), contact_s))
            story.append(HRFlowable(width="100%", thickness=1.1,
                                    color=DARK, spaceBefore=2, spaceAfter=4))
            contact_done = True
            i += 1
            continue

        # Section heading detection
        upper = line.upper().strip()
        if upper in SECTIONS:
            story.append(Paragraph(line.upper(), sec_s))
            story.append(hr())
            current_section = upper
            i += 1
            continue

        # TECHNICAL SKILLS section
        if current_section == "TECHNICAL SKILLS":
            if ':' in line:
                parts = line.split(':', 1)
                label = safe(parts[0].strip())
                val = safe(parts[1].strip())
                story.append(Paragraph(f"<b>{label}:</b>  {val}", skill_s))
            i += 1
            continue

        # PROJECTS section
        if current_section == "PROJECTS":
            # Bullet
            if line.startswith('•') or line.startswith('-'):
                clean = re.sub(r'^[•\-]\s*', '', line)
                story.append(b(clean))
                i += 1
                continue
            # Live demo
            if line.lower().startswith('live demo'):
                story.append(Paragraph(f"&#x2022; {safe(line)}", bul_s))
                i += 1
                continue
            # Stack line (contains ·)
            if '·' in line:
                story.append(Paragraph(safe(line), stack_s))
                i += 1
                continue
            # Project title — bold, short, capitalized
            if len(line) < 70 and line[0].isupper():
                story.append(Paragraph(safe(line), proj_s))
                i += 1
                continue
            # Body fallback
            story.append(Paragraph(safe(line), body_s))
            i += 1
            continue

        # CERTIFICATIONS section
        if current_section == "CERTIFICATIONS":
            if line.startswith('•') or line.startswith('-'):
                clean = re.sub(r'^[•\-]\s*', '', line)
                story.append(b(clean))
            else:
                story.append(Paragraph(safe(line), body_s))
            i += 1
            continue

        # EDUCATION section
        if current_section == "EDUCATION":
            if line[0].isupper():
                story.append(Paragraph(safe(line), proj_s))
            else:
                story.append(Paragraph(safe(line), body_s))
            i += 1
            continue

        # PROFESSIONAL SUMMARY
        if current_section == "PROFESSIONAL SUMMARY":
            story.append(Paragraph(safe(line), body_s))
            i += 1
            continue

        # Default
        if line.startswith('•') or line.startswith('-'):
            clean = re.sub(r'^[•\-]\s*', '', line)
            story.append(b(clean))
        else:
            story.append(Paragraph(safe(line), body_s))
        i += 1

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

                score = res_json.get("ATS Score", "N/A")
                st.metric(label="🎯 ATS Compatibility Score", value=score)

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

                st.subheader("💡 Expert Suggestions")
                suggestions = res_json.get("Suggestions", [])
                if suggestions:
                    for i, sug in enumerate(suggestions, 1):
                        st.info(f"**{i}.** {sug}")
                else:
                    st.info("No additional suggestions.")

                st.markdown("---")

                # ✨ Optimized Resume Download
                st.markdown("## ✨ Download Optimized Resume")
                st.markdown("AI rewrites your resume with missing keywords naturally added — same structure, same background.")

                with st.spinner("✍️ Generating optimized resume..."):
                    try:
                        optimized_text = get_optimized_resume_text(
                            resume_text,
                            jd_context,
                            keywords if keywords else [],
                            suggestions if suggestions else []
                        )

                        pdf_buffer = generate_optimized_pdf(optimized_text)

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
                        st.caption("💡 Same content — keywords and suggestions naturally added.")

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
