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
import io

# --- Configuration ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def clean_json_response(text):
    if not text or not text.strip():
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    return text

def get_working_model():
    preferred = ["gemini-1.5-flash-latest", "gemini-1.5-flash-001", "gemini-1.5-pro-latest"]
    try:
        available = [m.name.replace("models/", "") for m in genai.list_models()
                     if "generateContent" in m.supported_generation_methods]
        for p in preferred:
            if p in available:
                return p
        if available:
            return available[0]
    except Exception:
        pass
    return "gemini-1.5-flash-latest"

def get_ai_response(input_text, pdf_content, prompt):
    full_prompt = f"Job Context: {input_text}\n\nResume:\n{pdf_content}\n\nInstructions:\n{prompt}"
    model_name = get_working_model()
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(full_prompt)
        if not response.candidates:
            return "ERROR: No response"
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def get_optimized_resume_text(resume_text, jd_context, missing_keywords, suggestions):
    prompt = f"""
You are an expert resume writer. Reorganize and rewrite the following resume text line-by-line into a highly optimized format that follows this exact structural flow:

1. SECTION ORDER:
   - PROFESSIONAL SUMMARY
   - EDUCATION
   - CERTIFICATIONS
   - TECHNICAL SKILLS
   - PROJECTS

2. CLEAN LINK INTEGRATION:
   - Do NOT output raw or ugly URL bracket texts like '[Live Demo: ...]' or '[View Certificate]'.
   - Format all hyperlinks natively or embed them directly onto the anchor words. For instance, the Project Title text should be the hyperlinked text if a link is present.

3. CLEANUP CONTENT:
   - Remove trailing raw date text brackets from titles such as '()', '(Mar-)', or '(Jul-)'.
   - Fix typos like 'nan' to 'n8n' under projects.

Job Description Context: {jd_context}
Missing Keywords to inject: {', '.join(missing_keywords)}
Suggestions to adapt: {'; '.join(suggestions)}

Original Resume Data:
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

def format_live_links(text):
    """ Converts text containing bracketed or plain markdown links into clear, clickable HTML anchor syntax. """
    # Strip explicit raw brackets first
    text = re.sub(r'\[Live Demo:\s*([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[View Certificate\]', '', text)
    
    url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|(?:github\.com|linkedin\.com|streamlit\.app)/[^\s<>"]*)'
    def replace_url(match):
        url = match.group(0)
        href = url if url.startswith(('http://', 'https://')) else 'https://' + url
        clean_name = url.replace('https://', '').replace('http://', '').replace('www.', '')
        return f'<a href="{href}" color="#1a56db"><b>{clean_name}</b></a>'
    
    return re.sub(url_pattern, replace_url, text)

def generate_optimized_pdf(optimized_text):
    buffer = io.BytesIO()
    lines = [l.strip() for l in optimized_text.split('\n') if l.strip()]

    # Precise structural layout bounds tracking target margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=38,
        bottomMargin=38
    )

    DARK = colors.HexColor("#111111")
    BLUE = colors.HexColor("#1a56db")
    GRAY = colors.HexColor("#2d3748")

    def Style(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9.5, textColor=GRAY, leading=13, spaceAfter=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    name_s    = Style("name", fontName="Helvetica-Bold", fontSize=18, textColor=DARK, alignment=TA_CENTER, spaceAfter=4, leading=22)
    contact_s = Style("contact", fontSize=8.5, textColor=GRAY, alignment=TA_CENTER, spaceAfter=10)
    sec_s     = Style("sec", fontName="Helvetica", fontSize=11, textColor=DARK, spaceBefore=10, spaceAfter=2)
    proj_s    = Style("proj", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, spaceBefore=5, spaceAfter=1)
    body_s    = Style("body", fontSize=9.5, textColor=GRAY, leading=13, spaceAfter=2, alignment=TA_JUSTIFY)
    bul_s     = Style("bul", fontSize=9.5, textColor=GRAY, leading=13, leftIndent=12, firstLineIndent=-12, spaceAfter=2, alignment=TA_JUSTIFY)
    cert_l_s  = Style("cert_l", fontName="Helvetica-Bold", fontSize=9.5, textColor=DARK, alignment=TA_LEFT)
    cert_r_s  = Style("cert_r", fontName="Helvetica-Oblique", fontSize=9.0, textColor=DARK, alignment=TA_RIGHT)

    SECTIONS = ["PROFESSIONAL SUMMARY", "EDUCATION", "CERTIFICATIONS", "TECHNICAL SKILLS", "PROJECTS"]

    def hr():
        return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#4a5568"), spaceBefore=1, spaceAfter=5)

    def safe(t):
        return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    story = []
    name_done, contact_done = False, False
    current_section = None

    for line in lines:
        line = re.sub(r'\*\*', '', line)
        
        if not name_done:
            story.append(Paragraph(safe(line), name_s))
            name_done = True
            continue
            
        if not contact_done:
            # Inline contact metadata tracking icons format setup
            story.append(Paragraph(format_live_links(safe(line)), contact_s))
            contact_done = True
            continue

        upper = line.upper().rstrip('.')
        if upper in SECTIONS:
            story.append(Paragraph(safe(line.upper()), sec_s))
            story.append(hr())
            current_section = upper
            continue

        # Dynamic Certification Row Parsing Setup
        if current_section == "CERTIFICATIONS":
            date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[\–\-]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})', line)
            if date_match:
                date_str = date_match.group(0)
                cert_title = line.replace(date_str, "").strip().rstrip("–- ")
                # Structural sanitization for artifacts
                cert_title = cert_title.replace("()", "").replace("(Mar-)", "").replace("(Jul-)", "").strip()
                
                t = Table([[Paragraph(safe(cert_title), cert_l_s), Paragraph(safe(date_str), cert_r_s)]], colWidths=[370, 140])
                t.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                    ('TOPPADDING', (0,0), (-1,-1), 3),
                ]))
                story.append(t)
                continue

        # Dynamic Projects Section Processing Engine
        if current_section == "PROJECTS":
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                clean = re.sub(r'^[•\-\*]\s*', '', line)
                story.append(Paragraph(f"•  {safe(clean)}", bul_s))
                continue
            elif len(line) < 120 and not line.endswith('.'):
                story.append(Paragraph(format_live_links(safe(line)), proj_s))
                continue

        # General Text Element Handlers
        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
            clean = re.sub(r'^[•\-\*]\s*', '', line)
            story.append(Paragraph(f"•  {safe(clean)}", bul_s))
        elif ':' in line and current_section == "TECHNICAL SKILLS":
            parts = line.split(':', 1)
            story.append(Paragraph(f"<b>{safe(parts[0].strip())}:</b> {safe(parts[1].strip())}", body_s))
        else:
            story.append(Paragraph(safe(line), body_s))

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

# --- Streamlit Presentation Layer ---
st.set_page_config(page_title="AI Resume Reviewer", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em;
        background-color: #ff4b4b; color: white; font-weight: bold; font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI Resume Reviewer")
st.subheader("ATS Scoring + Optimized Exact Match Design Layout")

with st.sidebar:
    st.header("⚙️ Target Profile Setting")
    jd_context = st.text_area("Target Job Role / Description", value="Data Science & Analytics Internship Role", height=150)

uploaded_file = st.file_uploader("Upload Current Resume PDF", type=["pdf"])
submit = st.button("🔍 Run Full Synchronization Engine", use_container_width=True)

if submit:
    if uploaded_file is None:
        st.error("❌ Please upload a resume PDF.")
    else:
        with st.spinner("Analyzing and synchronizing layouts..."):
            try:
                resume_text = input_pdf_text(uploaded_file)
                
                # Core analysis prompt execution
                analysis_prompt = "Return JSON with: 'ATS Score', 'Strengths', 'Weaknesses', 'Missing Keywords', 'Suggestions'"
                response_text = get_ai_response(jd_context, resume_text, analysis_prompt)
                res_json = json.loads(clean_json_response(response_text))

                st.markdown("## 📊 Synchronization Metrics")
                st.metric(label="🎯 Structural ATS Alignment Score", value=res_json.get("ATS Score", "N/A"))

                # Optimize and write binary buffer
                optimized_text = get_optimized_resume_text(
                    resume_text, 
                    jd_context, 
                    res_json.get("Missing Keywords", []), 
                    res_json.get("Suggestions", [])
                )
                
                pdf_buffer = generate_optimized_pdf(optimized_text)

                st.download_button(
                    label="📥 Download Exact Matching Target Layout Resume (PDF)",
                    data=pdf_buffer,
                    file_name=f"{uploaded_file.name.replace('.pdf','')}_synchronized.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Layout formatting completely synchronized to target rules! Links embedded and section orders swapped.")
            except Exception as e:
                st.error(f"❌ Execution Failure: {str(e)}")
