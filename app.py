import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.units import inch, cm
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
            return "ERROR: No response received."
        candidate = response.candidates[0]
        finish_reason = str(candidate.finish_reason)
        if finish_reason not in ("FinishReason.STOP", "1", "STOP"):
            return f"ERROR: Response blocked. Reason: {finish_reason}"
        if not candidate.content or not candidate.content.parts:
            return f"ERROR: Empty response."
        return candidate.content.parts[0].text
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def get_optimized_resume_text(resume_text, jd_context, missing_keywords, suggestions):
    prompt = f"""
You are an expert resume writer. Rewrite the following resume to achieve 90%+ ATS score.

STRICT RULES:
1. Keep EXACT same sections in this ORDER: Name, Contact, Professional Summary, Education, Certifications, Technical Skills, Projects
2. Keep ALL project names, certifications, education facts unchanged
3. Keep ALL live demo links exactly as they are
4. Incorporate these missing keywords naturally: {', '.join(missing_keywords)}
5. Apply these suggestions: {'; '.join(suggestions)}
6. Do NOT use markdown bold (**text**) — plain text only
7. Max 2 bullet points per project
8. Professional Summary: max 3 lines
9. Return ONLY resume text, no explanations

Job Description: {jd_context}

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

def safe(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def linkify(text):
    """Convert URLs to clickable blue links."""
    url_pattern = r'(https?://[^\s<>"]+|(?:github\.com|linkedin\.com|streamlit\.app|wellfound\.com)[^\s<>"]*)'
    def replace_url(match):
        url = match.group(0)
        href = url if url.startswith('http') else f'https://{url}'
        display = url.replace('https://','').replace('http://','')
        return f'<a href="{href}" color="#2563EB"><u>{safe(display)}</u></a>'
    return re.sub(url_pattern, replace_url, text)

def generate_optimized_pdf(optimized_text):
    buffer = io.BytesIO()

    PAGE_W, PAGE_H = A4
    MARGIN = 1.8 * cm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    # Colors — matching Image 2
    BLACK  = colors.HexColor("#111827")
    BLUE   = colors.HexColor("#2563EB")
    GRAY   = colors.HexColor("#374151")
    LGRAY  = colors.HexColor("#6B7280")
    DIVIDER= colors.HexColor("#D1D5DB")

    BF = 8.8   # base font
    BL = 12.0  # base leading

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=BF, textColor=GRAY,
                    leading=BL, spaceAfter=0, spaceBefore=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    # Styles — matching Image 2 exactly
    name_s    = S("nm", fontName="Helvetica-Bold", fontSize=20, textColor=BLACK,
                  alignment=TA_CENTER, spaceAfter=4, leading=24)
    contact_s = S("ct", fontSize=8, textColor=LGRAY,
                  alignment=TA_CENTER, spaceAfter=2, leading=11)
    sec_s     = S("sc", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLACK,
                  spaceBefore=10, spaceAfter=3, leading=12)
    body_s    = S("bd", fontSize=BF, textColor=GRAY,
                  leading=BL, spaceAfter=2, alignment=TA_JUSTIFY)
    bul_s     = S("bl", fontSize=BF, textColor=GRAY, leading=BL,
                  leftIndent=12, firstLineIndent=-12, spaceAfter=2)
    proj_s    = S("pj", fontName="Helvetica-Bold", fontSize=BF+0.3, textColor=BLUE,
                  spaceBefore=6, spaceAfter=1)
    stack_s   = S("st", fontSize=BF-0.5, textColor=LGRAY, spaceAfter=2)
    skill_s   = S("sk", fontSize=BF, textColor=GRAY, leading=BL, spaceAfter=2)
    cert_name_s = S("cn", fontName="Helvetica-Bold", fontSize=BF, textColor=BLACK,
                    spaceAfter=0, spaceBefore=3)
    cert_sub_s  = S("cs", fontSize=BF-0.5, textColor=LGRAY, spaceAfter=4, fontName="Helvetica-Oblique")
    edu_name_s  = S("en", fontName="Helvetica-Bold", fontSize=BF+0.5, textColor=BLACK,
                    spaceAfter=1, spaceBefore=3)
    edu_sub_s   = S("es", fontSize=BF-0.5, textColor=LGRAY, fontName="Helvetica-Oblique", spaceAfter=2)
    link_s      = S("lk", fontSize=BF, textColor=BLUE, leading=BL, spaceAfter=2,
                    leftIndent=12, firstLineIndent=-12)

    def hr(thick=0.5, color=DIVIDER, before=1, after=4):
        return HRFlowable(width="100%", thickness=thick, color=color,
                          spaceBefore=before, spaceAfter=after)

    def bullet(text):
        processed = linkify(safe(text))
        return Paragraph(f"• {processed}", bul_s)

    SECTIONS = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROJECTS",
        "CERTIFICATIONS", "EDUCATION", "EXPERIENCE", "ACHIEVEMENTS",
        "SKILLS", "SUMMARY"
    ]

    story = []
    lines = [l.strip() for l in optimized_text.split('\n')]
    # Remove markdown bold
    lines = [re.sub(r'\*\*', '', l) for l in lines]

    name_done = False
    contact_done = False
    current_section = None
    i = 0

    # Buffer for certifications (name + issuer + date on same row)
    cert_buffer = []

    def flush_certs():
        nonlocal cert_buffer
        for cb in cert_buffer:
            story.append(Paragraph(safe(cb['name']), cert_name_s))
            if cb.get('sub'):
                story.append(Paragraph(safe(cb['sub']), cert_sub_s))
        cert_buffer = []

    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        # Name
        if not name_done:
            story.append(Paragraph(safe(line), name_s))
            name_done = True
            i += 1
            continue

        # Contact line — icons style like Image 2
        if not contact_done:
            # Build contact with separators
            parts = [p.strip() for p in re.split(r'\s*[\|•]\s*', line)]
            contact_parts = []
            for part in parts:
                if not part:
                    continue
                # Check if URL
                if any(x in part.lower() for x in ['github', 'linkedin', 'streamlit', 'wellfound', 'http']):
                    url = part if part.startswith('http') else f'https://{part}'
                    label = part.replace('https://','').replace('http://','')
                    contact_parts.append(f'<a href="{url}" color="#2563EB">{safe(label)}</a>')
                else:
                    contact_parts.append(safe(part))
            story.append(Paragraph('   ·   '.join(contact_parts), contact_s))
            story.append(hr(thick=1.0, color=BLACK, before=4, after=6))
            contact_done = True
            i += 1
            continue

        # Section heading
        upper = line.upper().strip().rstrip('.')
        if upper in SECTIONS:
            if current_section == "CERTIFICATIONS":
                flush_certs()
            story.append(Paragraph(line.upper(), sec_s))
            story.append(hr(thick=0.5, color=DIVIDER, before=1, after=4))
            current_section = upper
            i += 1
            continue

        # PROFESSIONAL SUMMARY
        if current_section == "PROFESSIONAL SUMMARY":
            story.append(Paragraph(linkify(safe(line)), body_s))
            i += 1
            continue

        # EDUCATION
        if current_section == "EDUCATION":
            if line.startswith('•') or line.startswith('-'):
                clean = re.sub(r'^[•\-]\s*', '', line)
                story.append(bullet(clean))
            elif any(x in line for x in ['B.E', 'B.Tech', 'Bachelor', 'Master', 'M.Tech', 'VTU', 'University', 'College', 'BLDEA']):
                story.append(Paragraph(linkify(safe(line)), edu_name_s))
            elif any(x in line for x in ['20', 'Data Science', 'Computer Science', 'Engineering']):
                story.append(Paragraph(safe(line), edu_sub_s))
            else:
                story.append(Paragraph(safe(line), body_s))
            i += 1
            continue

        # CERTIFICATIONS
        if current_section == "CERTIFICATIONS":
            clean = re.sub(r'^[•\-]\s*', '', line)
            # Try to detect cert name vs issuer/date
            if any(x in clean for x in ['Forage', 'NPTEL', 'be10x', 'Coursera', 'Google', 'Microsoft', 'AWS', 'View']):
                if cert_buffer:
                    cert_buffer[-1]['sub'] = cert_buffer[-1].get('sub', '') + clean
            else:
                # New cert entry
                cert_buffer.append({'name': clean, 'sub': ''})
            i += 1
            continue

        # TECHNICAL SKILLS
        if current_section == "TECHNICAL SKILLS":
            if ':' in line:
                parts = line.split(':', 1)
                label = safe(parts[0].strip())
                val = linkify(safe(parts[1].strip()))
                story.append(Paragraph(f"<b>{label}:</b> {val}", skill_s))
            else:
                story.append(Paragraph(linkify(safe(line)), skill_s))
            i += 1
            continue

        # PROJECTS
        if current_section == "PROJECTS":
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                clean = re.sub(r'^[•\-\*]\s*', '', line)
                story.append(bullet(clean))
                i += 1
                continue

            if 'live demo' in line.lower() or 'streamlit.app' in line.lower():
                url_match = re.search(r'(https?://\S+|[\w\-]+\.streamlit\.app\S*)', line)
                if url_match:
                    url = url_match.group(1)
                    if not url.startswith('http'):
                        url = f'https://{url}'
                    display = url.replace('https://','')
                    story.append(Paragraph(
                        f'• <a href="{url}" color="#2563EB"><u>{safe(display)}</u></a>',
                        link_s
                    ))
                i += 1
                continue

            if '·' in line or ('Python' in line and not line.endswith('.')):
                story.append(Paragraph(safe(line), stack_s))
                i += 1
                continue

            if len(line) < 80 and line[0].isupper():
                story.append(Paragraph(linkify(safe(line)), proj_s))
                i += 1
                continue

            story.append(Paragraph(linkify(safe(line)), body_s))
            i += 1
            continue

        # Default
        if line.startswith('•') or line.startswith('-'):
            clean = re.sub(r'^[•\-]\s*', '', line)
            story.append(bullet(clean))
        else:
            story.append(Paragraph(linkify(safe(line)), body_s))
        i += 1

    # Flush remaining certs
    if current_section == "CERTIFICATIONS":
        flush_certs()

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

st.title("🚀 AI Resume Reviewer")
st.subheader("ATS Scoring + Optimized Resume Download powered by Google Gemini")
st.write("Boost your resume impact and get placement ready!")

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

st.markdown("### 📄 Upload Your Resume")
uploaded_file = st.file_uploader(
    "Upload in PDF format", type=["pdf"],
    help="Only PDF files are supported."
)

if uploaded_file:
    st.success(f"✅ File received: **{uploaded_file.name}**")

st.markdown("---")
submit = st.button("🔍 Analyze My Resume", use_container_width=True)

if submit:
    if uploaded_file is None:
        st.error("❌ Please upload your resume PDF first!")
    else:
        with st.spinner("🤖 AI is scanning your resume... please wait!"):
            response_text = ""
            try:
                resume_text = input_pdf_text(uploaded_file)

                if not resume_text.strip():
                    st.error("❌ Could not extract text from the PDF.")
                    st.stop()

                st.info(f"📃 Extracted **{len(resume_text)} characters** from resume.")

                response_text = get_ai_response(jd_context, resume_text, input_prompt)

                with st.expander("🛠️ Raw API Response (Debug)", expanded=False):
                    st.code(response_text, language="json")

                if response_text.startswith("ERROR:") or response_text.startswith("EXCEPTION:"):
                    st.error(f"🚨 API Error: {response_text}")
                    st.stop()

                cleaned = clean_json_response(response_text)
                if not cleaned:
                    st.error("❌ Empty response.")
                    st.stop()

                res_json = json.loads(cleaned)

                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                score = res_json.get("ATS Score", "N/A")
                st.metric(label="🎯 ATS Compatibility Score", value=score)

                try:
                    score_val = int(score.replace("%", "").strip())
                    if score_val >= 80:
                        st.success("🟢 Excellent! Your resume is well-optimized.")
                    elif score_val >= 60:
                        st.warning("🟡 Average. Some improvements needed.")
                    else:
                        st.error("🔴 Low score. Resume needs improvement.")
                except:
                    pass

                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Key Strengths")
                    for s in res_json.get("Strengths", []):
                        st.success(f"• {s}")

                with col2:
                    st.subheader("⚠️ Areas for Improvement")
                    for w in res_json.get("Weaknesses", []):
                        st.warning(f"• {w}")

                st.markdown("---")
                st.subheader("🔍 Missing Critical Keywords")
                keywords = res_json.get("Missing Keywords", [])
                if keywords:
                    kw_html = " ".join([
                        f'<span style="background:#ff4b4b;color:white;padding:4px 10px;'
                        f'border-radius:12px;margin:4px;display:inline-block;">{k}</span>'
                        for k in keywords
                    ])
                    st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 Resume is already keyword-rich!")

                st.markdown("---")
                st.subheader("💡 Expert Suggestions")
                suggestions = res_json.get("Suggestions", [])
                for idx, sug in enumerate(suggestions, 1):
                    st.info(f"**{idx}.** {sug}")

                st.markdown("---")

                st.markdown("## ✨ Download Optimized Resume")
                st.markdown("AI rewrites your resume to achieve **90%+ ATS score** — Image 2 style format, clickable links, 1 page.")

                with st.spinner("✍️ Generating optimized resume..."):
                    try:
                        optimized_text = get_optimized_resume_text(
                            resume_text, jd_context,
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
                        st.success("✅ Optimized resume ready!")
                        st.caption("💡 Clean format • Clickable links • 1 page • 90%+ ATS")

                    except Exception as e:
                        st.error(f"❌ Could not generate optimized resume: {str(e)}")
                        st.info("Your ATS analysis above is still complete.")

                st.markdown("---")
                st.caption("✨ Powered by Google Gemini | Built with Streamlit")

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON Parse Error: {e}")
                st.code(response_text)
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
