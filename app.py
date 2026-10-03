"""FormSaathi - Intelligent AI Civic Form Vision & Voice Assistant.

Core loop:
USER SHARES (Photo/Scan + Optional Voice Query)
-> MULTIMODAL AI ANALYZES (Gemma 4 / Gemini Multimodal Vision + Voice Phonetic Correction)
-> SMART RESPONSE (Plain-English fields, document readiness tracker, common pitfalls, steps)
-> APPLICANT ACTS (Check readiness, verify documents, voice read-aloud, download checklist)
"""

import hashlib
import io
import json
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# Vision & Voice Models
DEFAULT_VISION_MODEL = "gemini-3.8-flash"
AVAILABLE_MODELS = [
    "gemini-3.8-flash",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
]
VOICE_MODEL = "gemini-3.8-flash"
SAMPLE_JSON = "samples/sample_output.json"
SAMPLE_IMAGE = "samples/sample_form.jpg"
LANGUAGES = ["English", "Hindi", "Telugu", "Tamil", "Bengali", "Marathi", "Gujarati", "Kannada"]

PROMPT_TEMPLATE = """You are FormSaathi, an expert, patient civic assistant who simplifies official application and government forms for everyday citizens.

The provided image is a photo or scan of a form. Inspect it thoroughly and respond ONLY with a valid JSON object (no markdown formatting, no code fences, no extra commentary) in this exact structure:

{{
  "not_a_form": false,
  "form_name": "Official title of the form as printed on the document",
  "purpose": "Clear 1-2 sentence explanation in plain words of what this form accomplishes and who needs it",
  "fields": [
    {{
      "field": "Exact field label as printed on the form",
      "meaning": "Plain explanation of what specific information is being requested",
      "what_to_fill": "Exact, actionable instructions on how to fill it correctly (e.g., format like DD/MM/YYYY, BLOCK letters, valid IDs)"
    }}
  ],
  "documents_needed": [
    "Specific document name with brief criteria (e.g. Recent utility bill under 3 months)"
  ],
  "common_mistakes": [
    "Frequent pitfall that causes rejection or delay"
  ],
  "steps": [
    "Chronological step to complete and submit this form"
  ]
}}

Guidelines:
- Explain everything in clear, natural {LANGUAGE}. Keep official field labels recognizable.
- Base your analysis SOLELY on what is visible in the provided image. If a section is blurry, note it instead of guessing.
- Highlight up to 12 of the most critical fields applicants must get right.
- List ONLY supporting documents explicitly asked for or strictly required for this form.
- If the image is NOT an application form or civic document, return {{"not_a_form": true}} and nothing else.
- Additional user context & spoken criteria: {CONTEXT}
"""


# ---------------------------------------------------------------- Helpers
def get_client():
    """Retrieve Gemini client using GEMINI_API_KEY from environment."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def prepare_image(raw: bytes) -> bytes:
    """Standardize and optimize image size for fast multimodal processing."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_json(text: str) -> dict:
    """Extract valid JSON from model response even if surrounded by markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No valid JSON structure found in model response")
    return json.loads(text[start : end + 1])


def normalise(data: dict) -> dict:
    """Ensure all required keys exist to guarantee UI stability."""
    data.setdefault("not_a_form", False)
    data.setdefault("form_name", "Unidentified Form")
    data.setdefault("purpose", "No description provided.")
    data.setdefault("fields", [])
    data.setdefault("documents_needed", [])
    data.setdefault("common_mistakes", [])
    data.setdefault("steps", [])
    data["fields"] = [f for f in data["fields"] if isinstance(f, dict)]
    return data


def transcribe_and_correct_voice(client, audio_bytes: bytes) -> dict:
    """
    Transcribe spoken voice audio using Gemini and actively correct phonetic
    mishearings or misspoken Indian civic words (Aadhaar, EPIC, PAN, Affidavit,
    Tehsildar, Domicile, Gazetted officer, Ration card, etc.).
    """
    voice_prompt = """You are an expert voice transcription engine specialized in Indian civic and government application scenarios.
Listen to this audio carefully and transcribe what the user is saying.

CRITICAL INSTRUCTION - PHONETIC & WORDING CORRECTION:
Speech-to-text systems frequently mishear Indian names, accents, and civic terms. Detect any phonetic errors or acoustic slips and intelligently fix them to what the user intended in this context.
Examples of common mishearings to correct:
- 'other card' / 'adhar' -> 'Aadhaar card'
- 'epic card' / 'apex' -> 'EPIC / Voter ID'
- 'pan' / 'pen card' -> 'PAN card'
- 'after david' / 'david' -> 'affidavit'
- 'gadget officer' -> 'Gazetted officer'
- 'talk' / 'talluk' -> 'taluk / tehsil'
- 'ration' / 'rashan' -> 'Ration card'
- 'domicile' / 'domical' -> 'Domicile certificate'

Respond ONLY with a valid JSON object in this exact shape:
{
  "transcription": "The cleaned, intelligently corrected transcript of what the user meant",
  "raw_heard": "What was phonetically heard before correction",
  "corrections": ["List of any specific words corrected, e.g., 'other card -> Aadhaar card', or empty list if none needed"]
}
"""
    try:
        response = client.models.generate_content(
            model=VOICE_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                voice_prompt,
            ],
        )
        return extract_json(response.text)
    except Exception as e:
        return {
            "transcription": "",
            "raw_heard": "",
            "corrections": [f"Voice transcription note: {e}"],
        }


def ask_vision_model(client, model_name: str, image_bytes: bytes, language: str, context: str) -> dict:
    """Call vision model with retry handling."""
    prompt = PROMPT_TEMPLATE.replace("{LANGUAGE}", language).replace(
        "{CONTEXT}", context.strip() if context else "None provided"
    )
    last_err = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            return normalise(extract_json(response.text))
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(str(last_err))


def build_checklist_text(data: dict, have: list) -> str:
    """Generate clean plain-text checklist for download."""
    lines = [
        "=" * 60,
        f"FORMSAATHI APPLICATION CHECKLIST: {data.get('form_name', 'Form')}",
        "=" * 60,
        "",
        f"PURPOSE:\n{data.get('purpose', '')}",
        "",
        "-" * 40,
        "DOCUMENT READINESS STATUS",
        "-" * 40,
    ]
    for d in data.get("documents_needed", []):
        status = "[READY]" if d in have else "[PENDING]"
        lines.append(f"{status} {d}")

    lines += [
        "",
        "-" * 40,
        "KEY FIELDS TO COMPLETE CAREFULLY",
        "-" * 40,
    ]
    for idx, f in enumerate(data.get("fields", []), 1):
        lines.append(f"{idx}. {f.get('field', 'Field')}")
        lines.append(f"   Meaning: {f.get('meaning', 'N/A')}")
        lines.append(f"   How to fill: {f.get('what_to_fill', 'N/A')}")

    lines += [
        "",
        "-" * 40,
        "COMMON MISTAKES TO AVOID",
        "-" * 40,
    ]
    for m in data.get("common_mistakes", []):
        lines.append(f"• {m}")

    lines += [
        "",
        "-" * 40,
        "SUBMISSION STEPS",
        "-" * 40,
    ]
    for i, s in enumerate(data.get("steps", []), 1):
        lines.append(f"{i}. {s}")

    lines += [
        "",
        "=" * 60,
        "Generated with FormSaathi.",
        "Disclaimer: AI guidance can vary. Always verify details with official portals.",
        "=" * 60,
    ]
    return "\n".join(lines)


def build_markdown_report(data: dict, have: list) -> str:
    """Generate formatted Markdown report."""
    md = [
        f"# FormSaathi Submission Guide: {data.get('form_name', 'Form')}",
        f"\n**Purpose:** {data.get('purpose', '')}\n",
        "## Document Readiness Checklist",
    ]
    for d in data.get("documents_needed", []):
        check = "x" if d in have else " "
        md.append(f"- [{check}] {d}")

    md.append("\n## Key Form Fields Guidance")
    for f in data.get("fields", []):
        md.append(f"### {f.get('field', 'Field')}")
        md.append(f"- **Meaning:** {f.get('meaning', 'N/A')}")
        md.append(f"- **How to Fill:** {f.get('what_to_fill', 'N/A')}")

    md.append("\n## Common Pitfalls & Mistakes")
    for m in data.get("common_mistakes", []):
        md.append(f"- {m}")

    md.append("\n## Step-by-Step Submission Roadmap")
    for i, s in enumerate(data.get("steps", []), 1):
        md.append(f"{i}. {s}")

    md.append("\n---\n*Generated by FormSaathi.*")
    return "\n".join(md)


def load_sample():
    """Load default sample demonstration JSON."""
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        return normalise(json.load(f))


# ---------------------------------------------------------------- Modern UI Styling
st.set_page_config(
    page_title="FormSaathi | AI Civic Form & Voice Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek Modern Top Navbar (Replaces bulky hero box) */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 0 1.2rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-icon {
        background: linear-gradient(135deg, #4F46E5, #3B82F6);
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }
    
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .brand-sub {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
    }
    
    .nav-chips {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .chip {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    
    .chip.primary {
        background: #EEF2FF;
        color: #4F46E5;
        border-color: #C7D2FE;
    }
    
    /* Modern Section Header */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Voice Assistant Card */
    .voice-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    
    .voice-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #4F46E5;
        background: #EEF2FF;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    .correction-alert {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-top: 0.6rem;
        font-size: 0.86rem;
        color: #065F46;
    }
    
    /* Pitfall card */
    .pitfall-item {
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
        border-radius: 6px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.6rem;
        color: #92400E;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    
    /* Step Roadmap */
    .step-card {
        display: flex;
        gap: 12px;
        padding: 0.7rem 0;
        border-bottom: 1px dashed #E2E8F0;
        align-items: flex-start;
    }
    .step-card:last-child {
        border-bottom: none;
    }
    .step-badge {
        background: #4F46E5;
        color: white;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        flex-shrink: 0;
    }
    
    /* Primary buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Sleek Top Navbar
st.markdown(
    """
    <div class="top-nav">
        <div class="nav-brand">
            <div class="brand-icon">📄</div>
            <div>
                <div class="brand-title">FormSaathi</div>
                <div class="brand-sub">AI Vision & Voice Assistant for Government & Application Forms</div>
            </div>
        </div>
        <div class="nav-chips">
            <span class="chip primary">⚡ Multimodal Vision</span>
            <span class="chip">🎙️ Voice Enabled</span>
            <span class="chip">🛡️ Privacy First</span>
            <span class="chip">📋 Smart Readiness</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Assistant Settings")

    selected_model = st.selectbox(
        "Vision & Analysis Engine",
        AVAILABLE_MODELS,
        index=0,
        help="Select the AI model for visual form decoding.",
    )

    language = st.selectbox(
        "Response Language",
        LANGUAGES,
        index=0,
        help="Select your preferred language for explanations.",
    )

    st.markdown("---")
    st.markdown("#### 📖 How It Works")
    st.markdown(
        """
        1. **Upload or Capture:** Snap a clear photo of an official form.
        2. **Speak Your Criteria (Voice):** Record what you need help with. AI auto-corrects misheard civic words.
        3. **Analyze:** Receive plain-English field breakdowns, mistake alerts, and a document readiness meter.
        """
    )

    st.markdown("---")
    st.markdown("#### 🛡️ Privacy Tip")
    st.caption("Use blank form templates or cover Aadhaar/PAN numbers when demonstrating.")

    show_raw = st.checkbox("Developer: Raw JSON Inspector", value=False)
    st.caption("FormSaathi • Civic Tech")

# ---------------------------------------------------------------- Main Workspace
col_input, col_preview = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="section-title"><span>📸</span> Step 1: Provide Form Photo</div>', unsafe_allow_html=True)

    tab_upload, tab_camera = st.tabs(["📁 File Upload", "📷 Live Camera"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload image (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

    with tab_camera:
        camera = st.camera_input("Capture form photo", label_visibility="collapsed")

    source = uploaded or camera

    # ------------------------------------------------------------ Voice Criteria & Speech Input
    st.markdown(
        '<div class="section-title" style="margin-top: 1rem;"><span>🎙️</span> Step 2: Voice Criteria & Context</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="voice-box">
            <span class="voice-badge">SMART VOICE RECOGNITION</span>
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.6rem;">
                Speak your questions or situation (e.g. <i>"I am applying for address change, my bill is in my father's name"</i>).
                AI automatically detects & fixes misheard civic words (e.g. <i>'other card'</i> → <i>'Aadhaar card'</i>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio recording input
    audio_record = st.audio_input("Record your query / instructions")

    # Handle voice transcription & mishearing correction
    if audio_record is not None:
        audio_bytes = audio_record.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if st.session_state.get("last_audio_hash") != audio_hash:
            client = get_client()
            if client:
                with st.spinner("Transcribing voice & fixing misheard terms..."):
                    voice_res = transcribe_and_correct_voice(client, audio_bytes)
                    st.session_state["last_audio_hash"] = audio_hash
                    st.session_state["transcribed_text"] = voice_res.get("transcription", "")
                    st.session_state["corrections_made"] = voice_res.get("corrections", [])
                    st.toast("Voice transcribed and refined!", icon="🎙️")

    # Show any automatic phonetic corrections
    if st.session_state.get("corrections_made"):
        corrections = st.session_state["corrections_made"]
        if corrections and any("->" in c or "→" in c for c in corrections):
            corrections_str = ", ".join(corrections)
            st.markdown(
                f"""
                <div class="correction-alert">
                    ✨ <b>Phonetic Correction Applied:</b> {corrections_str}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Editable Context Text Area (Allows user to inspect and adjust any words)
    default_context_value = st.session_state.get("transcribed_text", "")
    context = st.text_area(
        "Applicant Context & Specific Criteria (Editable):",
        value=default_context_value,
        placeholder="e.g. First-time applicant / Address change / Name spelling mismatch with ID proof...",
        help="Review or edit what was transcribed, or type manually.",
        height=80,
    )

    # Action Buttons
    c_btn1, c_btn2 = st.columns([1, 1], gap="medium")
    with c_btn1:
        run_analysis = st.button(
            "⚡ Analyze & Decode Form",
            type="primary",
            use_container_width=True,
            disabled=(source is None),
        )
    with c_btn2:
        run_sample = st.button(
            "✨ Try Sample Form (Demo)",
            use_container_width=True,
            help="Test with a realistic pre-analyzed official form without an API call.",
        )

with col_preview:
    st.markdown('<div class="section-title"><span>👁️</span> Form Image Preview</div>', unsafe_allow_html=True)
    if source is not None:
        st.image(source, caption="Your Uploaded Form", use_container_width=True)
    elif st.session_state.get("from_sample") and os.path.exists(SAMPLE_IMAGE):
        st.image(SAMPLE_IMAGE, caption="Demonstration Sample Form (Address Update)", use_container_width=True)
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #CBD5E1; border-radius: 12px; padding: 2.8rem 1.5rem; text-align: center; color: #64748B;">
                <p style="font-size: 2.2rem; margin-bottom: 0.5rem;">📄</p>
                <b style="color: #334155;">No document selected yet</b><br>
                <span style="font-size: 0.88rem;">Upload an image, take a photo, or click <b>'Try Sample Form'</b>.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- Execution & Model Call
if run_sample:
    try:
        st.session_state["data"] = load_sample()
        st.session_state["from_sample"] = True
        st.session_state["have_docs"] = []
        st.toast("Loaded sample demonstration form!", icon="✨")
    except FileNotFoundError:
        st.error(f"Sample data file not found at `{SAMPLE_JSON}`.")
        st.stop()

if run_analysis and source is not None:
    client = get_client()
    if client is None:
        st.error("🔑 **GEMINI_API_KEY is missing.** Please ensure it is defined in your `.env` file.")
        st.stop()

    try:
        image_bytes = prepare_image(source.getvalue())
    except Exception as e:
        st.error(f"Could not open this image: {e}")
        st.stop()

    cache_key = hashlib.md5(
        image_bytes + selected_model.encode() + language.encode() + (context or "").encode()
    ).hexdigest()
    cache = st.session_state.setdefault("cache", {})

    if cache_key in cache:
        st.session_state["data"] = cache[cache_key]
        st.toast("Retrieved result from cache.", icon="⚡")
    else:
        with st.spinner(f"FormSaathi is analyzing your form with {selected_model}..."):
            try:
                data = ask_vision_model(client, selected_model, image_bytes, language, context)
                cache[cache_key] = data
                st.session_state["data"] = data
            except Exception as e:
                st.error(f"Error from vision model: {e}")
                st.info("Check API rate limits or use 'Try Sample Form' to test immediately.", icon="💡")
                st.stop()
    st.session_state["from_sample"] = False
    st.session_state["have_docs"] = []

# ---------------------------------------------------------------- Results Section
data = st.session_state.get("data")
if not data:
    st.stop()

if data.get("not_a_form"):
    st.warning("⚠️ The uploaded image does not appear to be an application form or civic document. Please upload a clear photo of an official form.")
    st.stop()

st.markdown("---")

# Header & Overview
res_col1, res_col2 = st.columns([2.6, 1])
with res_col1:
    st.markdown(f"### 📋 {data.get('form_name', 'Application Form')}")
    st.markdown(f"**Purpose:** {data.get('purpose', 'N/A')}")
with res_col2:
    f_len = len(data.get("fields", []))
    d_len = len(data.get("documents_needed", []))
    m_len = len(data.get("common_mistakes", []))
    st.metric(label="Fields Identified", value=f"{f_len}")
    st.caption(f"Required Docs: **{d_len}** • Pitfalls: **{m_len}**")

# ---------------------------------------------------------------- Audio Read-Aloud (Voice Assistant Out)
# Creates a clean Web Speech read-aloud option in the browser
speech_text = f"Form: {data.get('form_name')}. Purpose: {data.get('purpose')}. Please have ready: {', '.join(data.get('documents_needed', []))}"
escaped_speech = speech_text.replace('"', '\\"').replace("'", "\\'")

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.8rem 0;">
        <span style="font-size: 1.2rem;">🔊</span>
        <div style="font-size: 0.88rem; color: #3730A3; flex-grow: 1;">
            <b>Audio Read-Aloud:</b> Listen to this form summary spoken aloud.
        </div>
        <button onclick="window.speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance('{escaped_speech}'); u.lang='en-IN'; window.speechSynthesis.speak(u);" 
                style="background: #4F46E5; color: white; border: none; border-radius: 6px; padding: 4px 12px; font-size: 0.82rem; font-weight: 600; cursor: pointer;">
            ▶ Play Audio
        </button>
        <button onclick="window.speechSynthesis.cancel();" 
                style="background: #E2E8F0; color: #334155; border: none; border-radius: 6px; padding: 4px 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer;">
            ⏹ Stop
        </button>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Document Readiness Meter
st.markdown("### ✅ Document Readiness Tracker")
docs = data.get("documents_needed", [])

if docs:
    have = st.multiselect(
        "Select the documents you currently have prepared:",
        options=docs,
        default=st.session_state.get("have_docs", []),
        key="have_docs",
    )

    total_docs = len(docs)
    prepared_docs = len(have)
    percentage = int((prepared_docs / total_docs) * 100) if total_docs > 0 else 100

    col_bar, col_status = st.columns([3, 1])
    with col_bar:
        st.progress(percentage / 100)
    with col_status:
        st.markdown(f"**Readiness:** `{prepared_docs}/{total_docs}` ({percentage}%)")

    missing = [d for d in docs if d not in have]

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        if have:
            st.success(f"**Ready for Submission ({len(have)}):**\n\n" + "\n".join([f"- ✅ {d}" for d in have]))
        else:
            st.info("No documents checked off yet. Check them above as you gather them.")

    with r_col2:
        if missing:
            st.error(f"**Still Needed ({len(missing)}):**\n\n" + "\n".join([f"- ⚠️ {d}" for d in missing]))
        elif have:
            st.balloons()
            st.success("🎉 **All required documents are ready for submission!**")
else:
    st.info("No explicit supporting documents required for this form.")
    have = []

# ---------------------------------------------------------------- Field Explorer
st.markdown("---")
f_head_col, f_search_col = st.columns([1.5, 1])
with f_head_col:
    st.markdown("### 📝 Field-by-Field Breakdown")
with f_search_col:
    search_query = st.text_input("Search fields...", placeholder="Type to filter fields...", label_visibility="collapsed")

fields = data.get("fields", [])
if search_query.strip():
    filtered_fields = [
        f for f in fields if search_query.lower() in f.get("field", "").lower() or search_query.lower() in f.get("meaning", "").lower()
    ]
else:
    filtered_fields = fields

if filtered_fields:
    for idx, f in enumerate(filtered_fields, 1):
        with st.expander(f"**{idx}. {f.get('field', 'Field')}**", expanded=(idx <= 3 and not search_query)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**📌 What it asks for:**\n\n{f.get('meaning', 'Not specified')}")
            with c2:
                st.markdown(f"**✍️ How to fill correctly:**\n\n{f.get('what_to_fill', 'Not specified')}")
else:
    st.info("No matching fields found.")

# ---------------------------------------------------------------- Pitfalls & Steps
st.markdown("---")
col_pitfalls, col_steps = st.columns(2, gap="large")

with col_pitfalls:
    st.markdown("### ⚠️ Common Mistakes to Avoid")
    mistakes = data.get("common_mistakes", [])
    if mistakes:
        for m in mistakes:
            st.markdown(f'<div class="pitfall-item"><b>•</b> {m}</div>', unsafe_allow_html=True)
    else:
        st.info("No common mistakes reported for this form.")

with col_steps:
    st.markdown("### 🚀 Step-by-Step Submission Roadmap")
    steps = data.get("steps", [])
    if steps:
        for i, s in enumerate(steps, 1):
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-badge">{i}</div>
                    <div style="color: #1E293B; line-height: 1.45; font-size: 0.92rem;">{s}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No steps reported.")

# ---------------------------------------------------------------- Export Tools
st.markdown("---")
st.markdown("### 📥 Export Submission Checklist")

d_col1, d_col2, d_col3 = st.columns([1, 1, 1])

checklist_text = build_checklist_text(data, have)
markdown_report = build_markdown_report(data, have)

with d_col1:
    st.download_button(
        "📄 Download Text Checklist (.txt)",
        data=checklist_text,
        file_name=f"{data.get('form_name', 'form').lower().replace(' ', '_')}_checklist.txt",
        mime="text/plain",
        use_container_width=True,
    )

with d_col2:
    st.download_button(
        "📑 Download Markdown Report (.md)",
        data=markdown_report,
        file_name=f"{data.get('form_name', 'form').lower().replace(' ', '_')}_guide.md",
        mime="text/markdown",
        use_container_width=True,
    )

with d_col3:
    if st.button("🖨️ View Printable Summary", use_container_width=True):
        st.session_state["show_print_view"] = not st.session_state.get("show_print_view", False)

if st.session_state.get("show_print_view"):
    st.code(checklist_text, language="text")

if show_raw:
    st.markdown("### 🛠️ Raw JSON")
    st.json(data)
