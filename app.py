import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import json
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import ParagraphStyle
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
You are an expert resume writer. Rewrite the following resume into a modern, highly optimized, ATS-friendly single-page resume layout following these specific requirements:

1. STRUCTURE & BACKGROUND (STRICT RETENTION):
   - Keep the EXACT same background context, experience details, project milestones, metrics, certifications, and academic background unchanged.
   - Do not change or invent factual career timelines or achievements.
   - Keep the exact section structures, order, and URLs as provided.

2. OPTIMIZATION:
   - Naturally incorporate these missing keywords without changing your actual background facts: {', '.join(missing_keywords)}
   - Address these professional suggestions: {'; '.join(suggestions)}
   - Eliminate redundant phrasing, low-impact adjectives, and wordy explanations.

3. DENSITY & SIZE RESTRICTIONS (STRICT ONE-PAGE TARGET):
   - Total length must NOT exceed 500 words.
   - The Professional Summary must be a maximum of 2 lines.
   - Include a maximum of 5 projects.
   - Restrict each project to a maximum of 2 highly impactful bullet points.
   - Trim repetitive bullet points across experience or accomplishments.

4. FORMATTING RULES:
   - Do NOT use markdown bold (**text**) inside the text bodies.
   - Return ONLY the clean resume text line-by-line, with absolutely no preamble, markdown code blocks, or conversational text.

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

def linkify_text(text):
    """
    Identifies HTTP/HTTPS links, GitHub, LinkedIn, and Streamlit domains
    and converts them into clickable anchor links matching the design precisely.
    """
    url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+|(?:github\.com|linkedin\.com|streamlit\.app)/[^\s<>"]*)'
    
    def replace_url(match):
        url = match.group(0)
        href = url
        if not url.startswith(('http://', 'https://')):
            href = 'https://' + url
        clean_url = url.replace('https://', '').replace('http://', '').replace('www.', '')
        return f'<a href="{href}" color="#1a56db"><u>{clean_url}</u></a>'
        
    return re.sub(url_pattern, replace_url, text)

def generate_optimized_pdf(optimized_text):
    buffer = io.BytesIO()

    # Pre-parse lines to track text complexity for Dynamic Spacing & Typography
    lines = [l.strip() for l in optimized_text.split('\n')]
    non_empty_lines = [l for l in lines if l]
    total_chars = sum(len(l) for l in non_empty_lines)

    # Smart Spacing Scaling Configurator
    if total_chars > 2000:
        base_font = 8.0
        base_leading = 10.5
        margin_x = 36  # 0.5 in
        margin_y = 28
        sec_space_before = 4
        sec_space_after = 1
    else:
        base_font = 8.5
        base_leading = 11.5
        margin_x = 36
        margin_y = 36  # 0.5 in
        sec_space_before = 5
        sec_space_after = 2

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=margin_x,
        leftMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y
    )

    DARK = colors.HexColor("#111111")
    BLUE = colors.HexColor("#1a56db")
    GRAY = colors.HexColor("#374151") 

    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=base_font,
                    textColor=GRAY, leading=base_leading, spaceAfter=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    name_s    = S("n", fontName="Helvetica-Bold", fontSize=base_font + 7, textColor=DARK,
                  alignment=TA_CENTER, spaceAfter=1, leading=base_leading + 6)
    contact_s = S("c", fontSize=base_font - 0.5, textColor=GRAY,
                  alignment=TA_CENTER, spaceAfter=3, leading=base_leading)
    sec_s     = S("sec", fontName="Helvetica-Bold", fontSize=base_font + 0.5, textColor=DARK,
                  spaceBefore=sec_space_before, spaceAfter=sec_space_after)
    proj_s    = S("proj", fontName="Helvetica-Bold", fontSize=base_font, textColor=DARK,
                  spaceBefore=sec_space_before - 2, spaceAfter=0)
    stack_s   = S("stack", fontName="Helvetica-Oblique", fontSize=base_font - 0.5,
                  textColor=BLUE, spaceAfter=1)
    body_s    = S("body", fontSize=base_font, textColor=GRAY,
                  leading=base_leading, spaceAfter=1, alignment=TA_JUSTIFY)
    bul_s     = S("bul", fontSize=base_font, textColor=GRAY, leading=base_leading,
                  leftIndent=14, firstLineIndent=-14, spaceAfter=1.5, alignment=TA_JUSTIFY)
    skill_s   = S("sk", fontSize=base_font, textColor=GRAY, leading=base_leading, spaceAfter=1)

    SECTIONS = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROJECTS",
        "CERTIFICATIONS", "EDUCATION", "EXPERIENCE", "ACHIEVEMENTS", "SKILLS", "SUMMARY"
    ]

    def hr():
        return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#D1D5DB"),
                          spaceBefore=1, spaceAfter=3)

    def safe(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    story = []
    name_done = False
    contact_done = False
    current_section = None
    
    project_counts = 0
    bullet_counts = 0

    for raw in lines:
        line = raw.strip()
        line = re.sub(r'\*\*', '', line)

        if not line:
            continue

        # Header Name Parsing
        if not name_done:
            story.append(Paragraph(linkify_text(safe(line)), name_s))
            name_done = True
            continue

        # Header Contact Metadata Parsing
        if not contact_done:
            story.append(Paragraph(linkify_text(safe(line)), contact_s))
            contact_done = True
            continue

        # Base Section Router
        upper = line.upper().rstrip('.')
        if upper in SECTIONS:
            story.append(Paragraph(safe(line.upper()), sec_s))
            story.append(hr())
            current_section = upper
            project_counts = 0
            continue

        # Technical Skills Rendering Flow
        if current_section == "TECHNICAL SKILLS":
            if ':' in line:
                parts = line.split(':', 1)
                label = safe(parts[0].strip())
                val = safe(parts[1].strip())
                story.append(Paragraph(f"<b>{label}:</b> {linkify_text(val)}", skill_s))
            else:
                story.append(Paragraph(linkify_text(safe(line)), skill_s))
            continue

        # Custom Project Aggregator Matching Layout
        if current_section == "PROJECTS":
            if line.startswith('•') or line.startswith('-') or line.startswith('■') or line.startswith('*'):
                if bullet_counts >= 2:
                    continue
                clean = re.sub(r'^[•\-\*■]\s*', '', line)
                story.append(Paragraph(f"&#9632;  {linkify_text(safe(clean))}", bul_s))
                bullet_counts += 1
                continue
            
            # Subtitle check (Contains space separation or classic indicators)
            if ('·' in line or '–' in line or '|' in line or line.startswith('Python') or line.startswith('n8n')) and not line.endswith('.'):
                formatted_stack = line.replace(' - ', ' · ').replace('   ', ' · ').replace('  ', ' · ')
                story.append(Paragraph(safe(formatted_stack), stack_s))
                continue
            
            # Project Header Title
            if len(line) < 80 and line[0].isupper() and not line.endswith('.'):
                if project_counts >= 5:
                    bullet_counts = 2
                    continue
                story.append(Paragraph(linkify_text(safe(line)), proj_s))
                project_counts += 1
                bullet_counts = 0
                continue

        # Generic Document Section Router Engine
        if line.startswith('•') or line.startswith('-') or line.startswith('■') or line.startswith('*'):
            clean = re.sub(r'^[•\-\*■]\s*', '', line)
            story.append(Paragraph(f"&#9632;  {linkify_text(safe(clean))}", bul_s))
        else:
            story.append(Paragraph(linkify_text(safe(line)), body_s))

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
