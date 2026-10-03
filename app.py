"""FormSaathi - photo of a government form -> Gemma 4 explains it -> checklist.

Core loop (Hack Day brief):
USER SHARES (photo) -> GEMMA UNDERSTANDS (gemma-4-31b-it via Gemini API)
-> APP RESPONDS (explained fields, documents, mistakes, steps)
-> USER ACTS (tick documents you have, see what's missing, download checklist)
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

MODEL = "gemma-4-31b-it"  # fallback option: "gemma-4-26b-a4b-it"
SAMPLE_JSON = "samples/sample_output.json"
SAMPLE_IMAGE = "samples/sample_form.jpg"  # optional: put a blank form photo here
LANGUAGES = ["English", "Hindi", "Telugu"]

PROMPT_TEMPLATE = """You are a helpful assistant who explains Indian government forms \
to first-time applicants.

The image is a photo or scan of a form. Read it carefully, then answer ONLY with a \
JSON object (no markdown, no extra text) in exactly this shape:

{
  "not_a_form": false,
  "form_name": "name of the form as printed on it",
  "purpose": "what this form is for, in 1-2 simple lines",
  "fields": [
    {"field": "field label as printed", "meaning": "what it is asking", "what_to_fill": "how to fill it correctly"}
  ],
  "documents_needed": ["document 1", "document 2"],
  "common_mistakes": ["mistake 1", "mistake 2"],
  "steps": ["step 1", "step 2", "step 3"]
}

Rules:
- Write all explanations in {LANGUAGE}. Use very simple words. Keep field labels and \
document names recognisable (you may keep the English label in brackets).
- Use ONLY what is visible in the image. If something is unclear or unreadable, say \
so in that item instead of guessing.
- List at most 12 of the most important fields.
- documents_needed: only documents the form itself asks for or clearly implies.
- If the image is NOT a government form or application form, return \
{"not_a_form": true} and nothing else.
- Extra context from the user: {CONTEXT}
"""


# ---------------------------------------------------------------- helpers
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return genai.Client(api_key=key)


def prepare_image(raw: bytes) -> bytes:
    """Convert to RGB JPEG and shrink big photos (faster + cheaper API call)."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, even if wrapped in ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in model reply")
    return json.loads(text[start : end + 1])


def normalise(data: dict) -> dict:
    """Make sure every key exists so the UI never crashes on a missing field."""
    data.setdefault("not_a_form", False)
    data.setdefault("form_name", "Unknown form")
    data.setdefault("purpose", "")
    data.setdefault("fields", [])
    data.setdefault("documents_needed", [])
    data.setdefault("common_mistakes", [])
    data.setdefault("steps", [])
    data["fields"] = [f for f in data["fields"] if isinstance(f, dict)]
    return data


def ask_gemma(client, image_bytes: bytes, language: str, context: str) -> dict:
    prompt = PROMPT_TEMPLATE.replace("{LANGUAGE}", language).replace(
        "{CONTEXT}", context or "none"
    )
    last_err = None
    for attempt in range(2):  # one retry for flaky network / bad JSON
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            return normalise(extract_json(response.text))
        except Exception as e:  # noqa: BLE001 - show any API/parse error nicely
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(str(last_err))


def build_checklist_text(data: dict, have: list) -> str:
    lines = [f"FORM: {data['form_name']}", "", f"PURPOSE: {data['purpose']}", ""]
    lines.append("DOCUMENTS")
    for d in data["documents_needed"]:
        lines.append(f"[{'x' if d in have else ' '}] {d}")
    lines += ["", "STEPS"]
    lines += [f"{i}. {s}" for i, s in enumerate(data["steps"], 1)]
    lines += ["", "COMMON MISTAKES TO AVOID"]
    lines += [f"- {m}" for m in data["common_mistakes"]]
    lines += ["", "Made with FormSaathi (Gemma 4). AI can make mistakes - verify on the official website before submitting."]
    return "\n".join(lines)


def load_sample():
    with open(SAMPLE_JSON, encoding="utf-8") as f:
        return normalise(json.load(f))


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="FormSaathi", page_icon="📄", layout="centered")
st.title("📄 FormSaathi")
st.caption("Sarkari form ki photo daalo. Gemma 4 samjhayega kya bharna hai, kaun se documents chahiye, aur kahan galti hoti hai.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Form ki photo upload karo ya camera se lo\n"
        "2. Gemma 4 form ko seedha *dekh ke* padhta hai (no separate OCR)\n"
        "3. Aapko explanation, document checklist aur steps milte hain"
    )
    st.info(f"Model: `{MODEL}` via Gemini API (open-weight Gemma 4)")
    st.warning("Privacy: asli Aadhaar/PAN ya personal data wali photo mat daalo. Blank form use karo.")
    show_raw = st.checkbox("Developer: show raw JSON")

language = st.selectbox("Explanation ki bhasha", LANGUAGES)
context = st.text_input(
    "Optional context",
    placeholder="e.g. Main pehli baar apply kar raha hoon / address change karna hai",
)

tab_up, tab_cam = st.tabs(["Upload photo", "Use camera"])
with tab_up:
    uploaded = st.file_uploader("Form ki photo", type=["png", "jpg", "jpeg", "webp"])
with tab_cam:
    camera = st.camera_input("Form ki photo lo")

source = uploaded or camera

col1, col2 = st.columns(2)
run = col1.button("Samjhao", type="primary", disabled=source is None)
sample = col2.button("Try sample (no API call)")

# ---- decide what to show
if sample:
    try:
        st.session_state["data"] = load_sample()
        st.session_state["from_sample"] = True
    except FileNotFoundError:
        st.error(f"{SAMPLE_JSON} nahi mila. Repo mein samples/ folder check karo.")
        st.stop()

if run and source is not None:
    client = get_client()
    if client is None:
        st.error("GEMINI_API_KEY set nahi hai. `.env` file mein add karo.")
        st.stop()

    try:
        image_bytes = prepare_image(source.getvalue())
    except Exception:  # noqa: BLE001
        st.error("Ye image open nahi ho paayi. Dusri photo try karo.")
        st.stop()

    cache_key = hashlib.md5(image_bytes + language.encode() + context.encode()).hexdigest()
    cache = st.session_state.setdefault("cache", {})

    if cache_key in cache:  # same input again -> no API call (saves rate limit)
        st.session_state["data"] = cache[cache_key]
    else:
        with st.spinner("Gemma 4 form padh raha hai..."):
            try:
                data = ask_gemma(client, image_bytes, language, context)
            except Exception as e:  # noqa: BLE001
                st.error(f"Gemma se jawab nahi aaya: {e}")
                st.info("Rate limit ho sakti hai. Thodi der baad try karo ya 'Try sample' dabao.")
                st.stop()
        cache[cache_key] = data
        st.session_state["data"] = data
    st.session_state["from_sample"] = False

data = st.session_state.get("data")
if not data:
    st.stop()

if data["not_a_form"]:
    st.warning("Ye image kisi sarkari form jaisi nahi lag rahi. Form ki saaf photo daalo.")
    st.stop()

# ---- show the result
if st.session_state.get("from_sample"):
    st.caption("Showing a pre-saved sample result.")
if source is not None and not st.session_state.get("from_sample"):
    st.image(source, caption="Aapka form", width=260)
elif st.session_state.get("from_sample") and os.path.exists(SAMPLE_IMAGE):
    st.image(SAMPLE_IMAGE, caption="Sample form", width=260)

st.header(data["form_name"])
st.write(data["purpose"])

st.subheader("Form mein kya poocha ja raha hai")
if data["fields"]:
    for f in data["fields"]:
        with st.expander(f.get("field", "Field")):
            st.markdown(f"**Matlab:** {f.get('meaning', '-')}")
            st.markdown(f"**Kya bharna hai:** {f.get('what_to_fill', '-')}")
else:
    st.write("Koi field samajh nahi aaya. Photo thodi aur saaf lo.")

# ---- second act: documents you have vs missing
st.subheader("Mere paas ye documents hain")
docs = data["documents_needed"]
have = st.multiselect("Jo documents ready hain wo chuno", docs, key="have_docs")
if docs:
    missing = [d for d in docs if d not in have]
    if have:
        st.success("Ready: " + ", ".join(have))
    if missing:
        st.error("Abhi chahiye: " + ", ".join(missing))
    elif have:
        st.balloons()
        st.success("Saare documents ready hain!")

st.subheader("Common galtiyan")
for m in data["common_mistakes"]:
    st.write(f"- {m}")

st.subheader("Steps")
for i, s in enumerate(data["steps"], 1):
    st.write(f"{i}. {s}")

st.download_button(
    "Checklist download karo",
    data=build_checklist_text(data, have),
    file_name="formsaathi_checklist.txt",
    mime="text/plain",
)

st.caption("AI se bana hai, galti ho sakti hai. Submit karne se pehle official website pe verify karein.")

if show_raw:
    st.json(data)
