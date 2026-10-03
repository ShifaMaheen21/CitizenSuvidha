# FormSaathi 📄

> **Simplify Any Civic or Government Application Form in Seconds.**  
> Built with **Gemma 4** (`gemma-4-31b-it`) via the Google Gemini API for Hacktoberfest Hack Day, Hyderabad (MLH × DEV × React Hyderabad).

---

## 🌟 Overview
Government and official application forms are often filled with obscure administrative jargon, strict formatting rules, and confusing document requirements. First-time applicants frequently face delays or outright rejections due to minor discrepancies (e.g., spelling mismatches, outdated utility bills, or signing outside borders).

**FormSaathi** is an AI-powered civic assistant. Upload a photo or capture an image of any official form, and **Gemma 4** visually parses it end-to-end—translating complex fields into clear, plain-English instructions, flagging common submission pitfalls, tracking document readiness, and generating downloadable checklists.

---

## ✨ Key Features
- **⚡ Direct Multimodal Vision (Gemma 4):** No separate, fragile OCR pipeline. Gemma 4 reads the layout, instructions, and fine print directly from images.
- **🇬🇧 100% Plain-English Guidance:** Clear, jargon-free explanations of what each field means and exactly how to fill it correctly.
- **✅ Interactive Document Readiness Tracker:** A real-time meter that calculates preparation percentage as you check off documents, showing what's ready and what's still missing.
- **🔍 Quick Field Search & Filter:** Instantly filter across long application forms to locate specific fields (e.g., *Address*, *Date of Birth*, *Signature*).
- **⚠️ Common Rejection Pitfalls:** Proactive warnings against common errors that cause application rejection.
- **📥 Multi-Format Action Exports:** Download your personalized application checklist as a `.txt` file, a comprehensive Markdown guide (`.md`), or view a printable summary card.
- **✨ Instant One-Click Demo:** Click **Try Sample Form** to test the complete workflow immediately with a realistic demonstration form without consuming API quota.

---

## 🔄 Core Loop
```text
USER SHARES PHOTO  ──►  GEMMA 4 UNDERSTANDS  ──►  APP EXPLAINS & TRACKS  ──►  USER ACTS & SUBMITS
(Upload / Camera)       (Direct Vision Model)     (Fields, Readiness, Traps)   (Ready Checklist & Steps)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ShifaMaheen21/FormSaathi.git
cd FormSaathi

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp env.example .env
```
Open `.env` in VS Code or your editor and set:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🛡️ Privacy & Safety Notice
- For live demonstrations, please use blank form templates or sample forms.
- Do not upload photos containing unmasked Aadhaar numbers, PAN numbers, or confidential financial details.
- Always cross-verify critical dates and fee structures on the official government department portal.

---

## 📜 License
MIT License. Built at Hacktoberfest Hack Day Hyderabad (MLH × DEV × React Hyderabad).
