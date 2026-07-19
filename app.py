import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
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
You are an expert resume writer. Rewrite the following resume text line-by-line into a highly optimized format that matches the target layout precisely:

1. STRUCTURE & BACKGROUND:
   - Keep the EXACT text details, structure sections, certifications, and educational data.
   - Do not invent factual data or dates. 
   
2. LINK HANDLING:
   - For projects that contain a Live Demo link (e.g. Streamlit app links), do NOT create a separate line saying "Live Demo: link". 
   - Instead, integrate the link directly into the Project Title text line by outputting it in brackets next to the title or embedding it so it reads cleanly as the title itself.

3. DENSITY TARGET:
   - Maximum 2 highly impactful lines for the summary.
   - Limit each project strictly to 2 clean bullet points starting with a standard dot (•). Do not include technology stack strings below titles.

Job Description Context: {jd_context}
Missing Keywords to inject: {', '.join(missing_keywords)}
Suggestions to adapt: {'; '.join(suggestions)}

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

def clean_and_format_links(text):
    """ Converts URLs or bracketed links inside text directly into bold blue hyperlinked text strings. """
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

    # Precise Letter/A4 margin configs matching image layout
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    DARK = colors.HexColor("#111111")
    BLUE = colors.HexColor("#1a56db")
    GRAY = colors.HexColor("#2d3748")

    def Style(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9.5, textColor=GRAY, leading=13, spaceAfter=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    name_s    = Style("name", fontName="Helvetica-Bold", fontSize=18, textColor=DARK, alignment=TA_CENTER, spaceAfter=4, leading=22)
    contact_s = Style("contact", fontSize=8.5, textColor=GRAY, alignment=TA_CENTER, spaceAfter=8)
    sec_s     = Style("sec", fontName="Helvetica", fontSize=11, textColor=DARK, spaceBefore=8, spaceAfter=2)
    proj_s    = Style("proj", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, spaceBefore=4, spaceAfter=1)
    body_s    = Style("body", fontSize=9.5, textColor=GRAY, leading=13, spaceAfter=2, alignment=TA_JUSTIFY)
    bul_s     = Style("bul", fontSize=9.5, textColor=GRAY, leading=13, leftIndent=12, firstLineIndent=-12, spaceAfter=2, alignment=TA_JUSTIFY)
    cert_l_s  = Style("cert_l", fontName="Helvetica-Bold", fontSize=9.5, textColor=DARK, alignment=TA_LEFT)
    cert_r_s  = Style("cert_r", fontName="Helvetica-Oblique", fontSize=9.0, textColor=DARK, alignment=TA_RIGHT)

    SECTIONS = ["PROFESSIONAL SUMMARY", "EDUCATION", "CERTIFICATIONS", "TECHNICAL SKILLS", "PROJECTS", "EXPERIENCE"]

    def hr():
        return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#4a5568"), spaceBefore=1, spaceAfter=4)

    def safe(t):
        return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    story = []
    name_done, contact_done = False, False
    current_section = None

    for line in lines:
        # Strip unexpected markdown artifacts
        line = re.sub(r'\*\*', '', line)
        
        if not name_done:
            story.append(Paragraph(safe(line), name_s))
            name_done = True
            continue
            
        if not contact_done:
            # Inline rendering for top metadata details row
            story.append(Paragraph(clean_and_format_links(safe(line)), contact_s))
            contact_done = True
            continue

        upper = line.upper().rstrip('.')
        if upper in SECTIONS:
            story.append(Paragraph(safe(line.upper()), sec_s))
            story.append(hr())
            current_section = upper
            continue

        # Certifications Section Right-Side Date Alignment Engine
        if current_section == "CERTIFICATIONS" and ("202" in line or "Jan" in line or "Feb" in line or "Mar" in line or "Apr" in line or "May" in line or "Jun" in line or "Jul" in line or "Aug" in line or "Sep" in line or "Oct" in line or "Nov" in line or "Dec" in line):
            date_match = re.search(r'([A-Za-z]+\s+\d{4}\s*[\–\-]\s*[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{4})', line)
            if date_match:
                date_str = date_match.group(0)
                cert_title = line.replace(date_str, "").strip().rstrip("–- ")
                
                # Render via an invisible border table tracking full width margins cleanly
                t = Table([[Paragraph(safe(cert_title), cert_l_s), Paragraph(safe(date_str), cert_r_s)]], colWidths=[370, 140])
                t.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                ]))
                story.append(t)
                continue

        # Project Section Link Integration Routing
        if current_section == "PROJECTS":
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                clean = re.sub(r'^[•\-\*]\s*', '', line)
                story.append(Paragraph(f"•  {safe(clean)}", bul_s))
                continue
            elif len(line) < 100 and not line.endswith('.'):
                # Apply high priority styling conversion to project title lines
                if "streamlit.app" in line or "github.com" in line:
                    story.append(Paragraph(clean_and_format_links(safe(line)), proj_s))
                else:
                    story.append(Paragraph(safe(line), proj_s))
                continue

        # Standard Text Elements Engine
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

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Job Settings")
    jd_context = st.text_area(
        "Target Job Role / Description",
        value="Software Engineer with skills in Python, Data Structures, REST APIs, and problem solving.",
        height=150
    )

# --- File Upload ---
st.markdown("### 📄 Upload Your Resume")
uploaded_file = st.file_uploader("Upload in PDF format", type=["pdf"])

st.markdown("---")
submit = st.button("🔍 Analyze My Resume", use_container_width=True)

if submit:
    if uploaded_file is None:
        st.error("❌ Please upload your resume PDF first!")
    else:
        with st.spinner("🤖 AI is scanning your resume... please wait!"):
            try:
                resume_text = input_pdf_text(uploaded_file)
                response_text = get_ai_response(jd_context, resume_text, input_prompt)
                cleaned = clean_json_response(response_text)
                res_json = json.loads(cleaned)

                # Display Results
                st.markdown("## 📊 Analysis Results")
                st.metric(label="🎯 ATS Compatibility Score", value=res_json.get("ATS Score", "N/A"))

                # ✨ Optimized Resume Download
                st.markdown("## ✨ Download Optimized Resume")
                
                optimized_text = get_optimized_resume_text(
                    resume_text,
                    jd_context,
                    res_json.get("Missing Keywords", []),
                    res_json.get("Suggestions", [])
                )

                pdf_buffer = generate_optimized_pdf(optimized_text)
                original_name = uploaded_file.name.replace(".pdf", "")

                st.download_button(
                    label="📥 Download Optimized Resume (PDF)",
                    data=pdf_buffer,
                    file_name=f"{original_name}_exact_match.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Optimized resume formatting matches the exact target layout! Click above to download.")

            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
