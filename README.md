# CitizenSuvidha

Take a photo of a government form. **Gemma 4** reads it directly and explains what each
field means, which documents you need, common mistakes, and the steps to apply.
Tick the documents you already have to see what's missing, then download a checklist.
Explanations come in English, Hindi or Telugu.

## The problem
Government forms are confusing, especially for first-time applicants. People make
small mistakes (wrong spelling, missing document) and get rejected or delayed.

## The Gemma 4 moment
`gemma-4-31b-it`, accessed through the Gemini API, looks at the photo of the form and
returns structured JSON (fields, documents, mistakes, steps) in the user's language.
No separate OCR, no templates. Remove Gemma and the app has nothing to show.

## Core loop
User shares photo -> Gemma 4 understands -> App explains + builds checklist -> User acts

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env      # then put your key from aistudio.google.com/apikey
streamlit run app.py
```

Click **Try sample** to see a pre-saved result without using the API.

## Notes
- Use blank forms for demos. Do not upload real personal documents.
- AI can make mistakes. Always verify on the official website before submitting.

## License
MIT (add the LICENSE file when creating the GitHub repo)

Built at Hacktoberfest Hack Day, Hyderabad (MLH x DEV x React Hyderabad).
