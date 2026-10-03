"""FormSaathi - sarkari forms, explained like a friend.

Warm, editorial, human-centered UI matching the exact FormSaathi design.
Core loop:
User shares form (Upload/Camera) + Optional Voice/Text details
-> AI decodes layout, fields, documents, pitfalls, and steps
-> User ticks documents, sees readiness, and downloads checklist.
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

PROMPT_TEMPLATE = """You are FormSaathi, a patient, warm, and highly knowledgeable friend who helps everyday citizens understand official government and civic application forms.

The image is a photo or scan of a form. Inspect it carefully and answer ONLY with a valid JSON object (no markdown code blocks, no backticks, no extra text) in this exact shape:

{{
  "not_a_form": false,
  "form_name": "Official title of the form as printed",
  "purpose": "A warm, plain-language 1-2 sentence explanation of what this form is for and why someone needs it",
  "fields": [
    {{
      "field": "Exact field label as printed",
      "meaning": "What this field is actually asking for in everyday terms",
      "what_to_fill": "Clear, friendly instructions on how to fill it correctly without mistakes (e.g. BLOCK letters, DD/MM/YYYY, matching ID proof)"
    }}
  ],
  "documents_needed": [
    "Exact document needed with any specific validity condition"
  ],
  "common_mistakes": [
    "Mistake that often gets this form rejected or delayed"
  ],
  "steps": [
    "Simple step to complete and submit this form"
  ]
}}

Rules:
- Write all explanations in natural {LANGUAGE}. Keep official field labels recognizable.
- Base your analysis solely on what is visible in the image.
- List up to 12 of the most essential fields to fill.
- Only include documents clearly required for this specific form.
- If the image is NOT an application form, return {{"not_a_form": true}} and nothing else.
- Extra context from the user: {CONTEXT}
"""


# ---------------------------------------------------------------- Helpers
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def prepare_image(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No valid JSON found in response")
    return json.loads(text[start : end + 1])


def normalise(data: dict) -> dict:
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
    Transcribes audio and intelligently detects and fixes misheard civic words,
    names, PIN codes, or administrative terms.
    """
    voice_prompt = """You are a smart voice listener for Indian civic application forms.
Transcribe what the user says.
CRITICAL: Voice mics often mishear Indian accents, names, numbers, PIN codes, and civic terms.
Correct any obvious phonetic slips:
- 'other card' / 'adhar' -> 'Aadhaar card'
- 'apex' / 'epic' -> 'EPIC / Voter ID'
- 'pen card' -> 'PAN card'
- 'after david' -> 'affidavit'
- 'gadget officer' -> 'Gazetted officer'
- 'talk' -> 'taluk / tehsil'
- 'rashan' -> 'Ration card'

Return ONLY valid JSON:
{
  "transcription": "The polished and corrected transcript",
  "raw_heard": "What was raw heard if different",
  "corrections": ["List of any corrected terms, or empty list"]
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
            "corrections": [f"Voice note: {e}"],
        }


def ask_vision_model(client, model_name: str, image_bytes: bytes, language: str, context: str) -> dict:
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
    lines = [
        f"FORM: {data.get('form_name', 'Form')}",
        f"PURPOSE: {data.get('purpose', '')}",
        "",
        "DOCUMENTS STATUS",
        "-" * 30,
    ]
    for d in data.get("documents_needed", []):
        status = "[x] READY:  " if d in have else "[ ] PENDING:"
        lines.append(f"{status} {d}")

    lines += ["", "KEY FIELDS TO FILL CAREFULLY", "-" * 30]
    for idx, f in enumerate(data.get("fields", []), 1):
        lines.append(f"{idx}. {f.get('field', 'Field')}")
        lines.append(f"   Meaning: {f.get('meaning', 'N/A')}")
        lines.append(f"   How to fill: {f.get('what_to_fill', 'N/A')}")

    lines += ["", "COMMON MISTAKES TO AVOID", "-" * 30]
    for m in data.get("common_mistakes", []):
        lines.append(f"• {m}")

    lines += ["", "NEXT STEPS", "-" * 30]
    for i, s in enumerate(data.get("steps", []), 1):
        lines.append(f"{i}. {s}")

    lines += ["", "Explained with FormSaathi • Always verify with the official government portal."]
    return "\n".join(lines)


def load_sample():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        return normalise(json.load(f))


# ---------------------------------------------------------------- Streamlit Page & Warm Aesthetic CSS
st.set_page_config(
    page_title="FormSaathi • sarkari forms, explained like a friend",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;0,6..72,800;1,6..72,400;1,6..72,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Overall warm page background */
    .stApp {
        background-color: #FAF6F0 !important;
        background-image: radial-gradient(#F0EAE1 1px, transparent 1px);
        background-size: 24px 24px;
        color: #292524 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Top Logo & Tagline */
    .brand-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
    }
    
    .brand-logo {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1C1917;
        letter-spacing: -0.03em;
    }
    
    .brand-tagline {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        color: #78716C;
        font-weight: 400;
    }
    
    /* Section labels */
    .field-heading {
        font-size: 0.95rem;
        font-weight: 600;
        color: #292524;
        margin-bottom: 0.4rem;
        margin-top: 1rem;
    }
    
    .field-caption {
        font-size: 0.86rem;
        color: #78716C;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    
    /* Start Here / Right Card Container */
    .start-here-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border: 1px dashed #D6CBB8;
        border-radius: 24px;
        padding: 2.8rem 2.4rem;
        box-shadow: 0 4px 20px -2px rgba(41, 37, 36, 0.04);
        margin-top: 0.5rem;
    }
    
    .start-badge {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #78716C;
        margin-bottom: 1.2rem;
    }
    
    .start-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #1C1917;
        line-height: 1.15;
        margin-bottom: 1.4rem;
        letter-spacing: -0.02em;
    }
    
    .start-desc {
        font-size: 1.02rem;
        color: #57534E;
        line-height: 1.65;
    }
    
    /* Soft alert box */
    .soft-alert {
        background-color: #FEE2E2;
        border: 1px solid #FECACA;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        color: #991B1B;
        font-size: 0.88rem;
        margin: 0.8rem 0;
    }
    
    .correction-badge {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        color: #065F46;
        font-size: 0.86rem;
        margin: 0.6rem 0;
    }
    
    /* Result card container */
    .result-container {
        background: #FFFFFF;
        border: 1px solid #E7DEC8;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(41, 37, 36, 0.05);
    }
    
    .result-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #1C1917;
        margin-bottom: 0.5rem;
    }
    
    .result-purpose {
        font-size: 1rem;
        color: #57534E;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }
    
    /* Custom button styling */
    div.stButton > button[kind="primary"] {
        background-color: #C25E38 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.65rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        box-shadow: 0 4px 12px rgba(194, 94, 56, 0.25) !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #B0502C !important;
        transform: translateY(-1px) !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #141C24 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.65rem 1.6rem !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #24303C !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }
    
    /* Multiselect and inputs styling */
    .stSelectbox, .stTextInput, .stTextArea {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Header
st.markdown(
    """
    <div class="brand-header">
        <span class="brand-logo">FormSaathi</span>
        <span class="brand-tagline">sarkari forms, explained like a friend</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Layout Columns
col_left, col_right = st.columns([1.1, 1.2], gap="large")

with col_left:
    # Language Selector
    st.markdown('<div class="field-heading" style="margin-top:0;">Explain it to me in</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Explain it to me in",
        LANGUAGES,
        index=0,
        label_visibility="collapsed",
    )

    # Tabs: Upload photo / Use camera
    tab_upload, tab_camera = st.tabs(["Upload photo", "Use camera"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            help="Supported formats: PNG, JPG, WEBP (up to 200MB)",
        )

    with tab_camera:
        camera = st.camera_input("Take photo", label_visibility="collapsed")

    source = uploaded or camera

    # Extra Details Section
    st.markdown('<div class="field-heading">Anything else we should know?</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode",
        ["Type it", "Speak it"],
        horizontal=True,
        index=1,
        label_visibility="collapsed",
    )

    client = get_client()

    if input_mode == "Speak it":
        st.markdown(
            '<div class="field-caption">Speak naturally. After recording, we\'ll show the words we heard — names, numbers and places often come out wrong.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="font-size: 0.9rem; font-weight: 600; color: #292524; margin-bottom: 4px;">Record a short note</div>', unsafe_allow_html=True)
        audio_record = st.audio_input("Record audio note", label_visibility="collapsed")

        if client is None:
            st.markdown(
                '<div class="soft-alert">Voice needs GEMINI_API_KEY in your .env file.</div>',
                unsafe_allow_html=True,
            )

        if audio_record is not None and client is not None:
            audio_bytes = audio_record.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()

            if st.session_state.get("last_audio_hash") != audio_hash:
                with st.spinner("Listening & checking names, PIN codes and wording..."):
                    voice_res = transcribe_and_correct_voice(client, audio_bytes)
                    st.session_state["last_audio_hash"] = audio_hash
                    st.session_state["transcribed_text"] = voice_res.get("transcription", "")
                    st.session_state["corrections_made"] = voice_res.get("corrections", [])

        # Display phonetic corrections badge if any
        if st.session_state.get("corrections_made"):
            corrs = [c for c in st.session_state["corrections_made"] if "->" in c or "→" in c]
            if corrs:
                st.markdown(
                    f'<div class="correction-badge">✨ <b>Corrected words:</b> {", ".join(corrs)}</div>',
                    unsafe_allow_html=True,
                )

        # Context text area for review
        context_val = st.session_state.get("transcribed_text", "")
        context = st.text_area(
            "What we heard (edit if needed):",
            value=context_val,
            placeholder="What you spoke will appear here. You can edit names, PIN codes or anything the mic got wrong.",
            height=70,
            label_visibility="visible" if context_val else "collapsed",
        )

    else:
        st.markdown(
            '<div class="field-caption">Mention any specific circumstances (e.g., first-time applicant, spelling mistake on old ID, address proof belongs to parents).</div>',
            unsafe_allow_html=True,
        )
        context = st.text_area(
            "Type any details:",
            placeholder="e.g. My father's name spelling is slightly different on my birth certificate...",
            height=85,
            label_visibility="collapsed",
        )

    # Action Buttons Row
    btn_col1, btn_col2 = st.columns([1, 1], gap="medium")
    with btn_col1:
        run_analysis = st.button(
            "Explain this form",
            type="primary",
            use_container_width=True,
            disabled=(source is None),
        )
    with btn_col2:
        run_sample = st.button(
            "Try a sample",
            type="secondary",
            use_container_width=True,
        )

# ---------------------------------------------------------------- Analysis Execution Logic
if run_sample:
    try:
        st.session_state["data"] = load_sample()
        st.session_state["from_sample"] = True
        st.session_state["have_docs"] = []
    except FileNotFoundError:
        st.error(f"Sample data file not found at `{SAMPLE_JSON}`.")
        st.stop()

if run_analysis and source is not None:
    if client is None:
        st.error("GEMINI_API_KEY is not configured in your `.env` file.")
        st.stop()

    try:
        image_bytes = prepare_image(source.getvalue())
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    cache_key = hashlib.md5(
        image_bytes + DEFAULT_VISION_MODEL.encode() + language.encode() + (context or "").encode()
    ).hexdigest()
    cache = st.session_state.setdefault("cache", {})

    if cache_key in cache:
        st.session_state["data"] = cache[cache_key]
    else:
        with st.spinner("FormSaathi is reading your form..."):
            try:
                data = ask_vision_model(client, DEFAULT_VISION_MODEL, image_bytes, language, context)
                cache[cache_key] = data
                st.session_state["data"] = data
            except Exception as e:
                st.error(f"Error reading form: {e}")
                st.stop()

    st.session_state["from_sample"] = False
    st.session_state["have_docs"] = []

# ---------------------------------------------------------------- Right Column Display
data = st.session_state.get("data")

with col_right:
    # Initial "Start Here" Card if no analysis has been run
    if not data:
        st.markdown(
            """
            <div class="start-here-card">
                <div class="start-badge">START HERE</div>
                <div class="start-title">Bring a form. We'll sit with you. 🔗</div>
                <div class="start-desc">
                    Upload a blank form photo, or try the sample. If you speak extra details, we'll show you the words we heard so you can fix names, PIN codes, and anything the mic got wrong.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Check if model flagged as not a form
        if data.get("not_a_form"):
            st.warning("⚠️ This image doesn't appear to be an application or government form. Please upload a clearer photo of an official form.")
        else:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)

            # Form Image Preview (Thumbnail)
            if source is not None and not st.session_state.get("from_sample"):
                st.image(source, caption="Your Uploaded Form", use_container_width=True)
            elif st.session_state.get("from_sample") and os.path.exists(SAMPLE_IMAGE):
                st.image(SAMPLE_IMAGE, caption="Sample Form (Address Update)", use_container_width=True)

            # Title & Purpose
            st.markdown(f'<div class="result-title">{data.get("form_name", "Application Form")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-purpose">{data.get("purpose", "")}</div>', unsafe_allow_html=True)

            # Audio Read-Aloud
            read_summary = f"{data.get('form_name')}. {data.get('purpose')}."
            clean_speech = read_summary.replace('"', '\\"').replace("'", "\\'")
            st.markdown(
                f"""
                <div style="margin-bottom: 1.2rem;">
                    <button onclick="window.speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance('{clean_speech}'); u.lang='en-IN'; window.speechSynthesis.speak(u);" 
                            style="background: #FAF6F0; border: 1px solid #D6CBB8; border-radius: 9999px; padding: 5px 14px; font-size: 0.85rem; font-weight: 600; color: #44403C; cursor: pointer;">
                        🔊 Listen to summary
                    </button>
                    <button onclick="window.speechSynthesis.cancel();" 
                            style="background: #FAF6F0; border: 1px solid #E7DEC8; border-radius: 9999px; padding: 5px 10px; font-size: 0.85rem; color: #78716C; cursor: pointer;">
                        ⏹ Stop
                    </button>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Document Checklist
            st.markdown("#### 📋 Documents You Need")
            docs = data.get("documents_needed", [])
            if docs:
                have = st.multiselect(
                    "Tick the documents you already have ready:",
                    options=docs,
                    default=st.session_state.get("have_docs", []),
                    key="have_docs",
                )
                missing = [d for d in docs if d not in have]

                if have and not missing:
                    st.success("🎉 All required documents are ready to go!")
                elif missing:
                    st.info(f"Still pending: **{len(missing)}** of {len(docs)} documents.")
            else:
                st.write("No specific supporting documents required.")
                have = []

            # Fields Explanations
            st.markdown("#### ✍️ How to Fill Each Field")
            fields = data.get("fields", [])
            for idx, f in enumerate(fields, 1):
                with st.expander(f"**{idx}. {f.get('field', 'Field')}**"):
                    st.markdown(f"**What it's asking:** {f.get('meaning', 'N/A')}")
                    st.markdown(f"**How to fill it:** {f.get('what_to_fill', 'N/A')}")

            # Common Mistakes
            mistakes = data.get("common_mistakes", [])
            if mistakes:
                st.markdown("#### ⚠️ Mistakes to Avoid")
                for m in mistakes:
                    st.markdown(f"• {m}")

            # Steps
            steps = data.get("steps", [])
            if steps:
                st.markdown("#### 🚀 Steps to Submit")
                for i, s in enumerate(steps, 1):
                    st.markdown(f"**{i}.** {s}")

            # Download Checklist
            st.markdown("---")
            st.download_button(
                "📥 Download Application Checklist (.txt)",
                data=build_checklist_text(data, have),
                file_name=f"{data.get('form_name', 'form').lower().replace(' ', '_')}_checklist.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown('</div>', unsafe_allow_html=True)
