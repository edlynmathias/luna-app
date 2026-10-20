# Luna: a gentle guide to periods

Luna is an AI agent for period-related questions, with two sections:

- **Symptom check.** Describe what you're feeling in your own words. Luna asks a couple of follow-up questions at a time as part of a normal conversation, remembers everything you've said, and then tells you which condition your symptoms sound closest to, with the NHS, MedlinePlus or ACOG pages behind it. You can stop to ask something ("how can I ease the pain?") and she picks up where she left off.
- **Ask about periods.** General questions, like how to ease cramps, what's normal for your cycle, or when to see a doctor, answered from trusted sources with citations. Follow-ups such as "what about at night?" work because Luna reads the whole conversation.

Educational only, not a diagnosis.

## Deploy on Streamlit Community Cloud

1. Put these files in a GitHub repository. They can sit at the top level or inside a folder.
2. At https://share.streamlit.io choose **Create app** and pick the repository.
3. Set **Main file path** to `streamlit_app.py`, or `your-folder/streamlit_app.py` if the files are in a folder.
4. Open **Advanced settings**, choose **Python 3.12**, and paste this into **Secrets**:
   ```
   LLM_API_KEY = "gsk_your_key_here"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
   A free Groq key is available at https://console.groq.com/keys. Any OpenAI-compatible provider works.
5. Click **Deploy**. The first start takes a few minutes while packages install and the small embedding model downloads.

When it's running, the sidebar footer should say "Search: semantic". If it says "keyword", `requirements.txt` wasn't found next to `streamlit_app.py`.

## Run locally

```
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your key
streamlit run streamlit_app.py
```

Without a key Luna still works, using keyword rules and source excerpts instead of AI-written replies.

## How it works

```
message + conversation so far + symptom picture
   1. Read      the AI picks out symptoms you have or don't have, and what you want (more questions, the result, or information)
   2. Score     assessment.py ranks 18 conditions against your symptoms (plain code, so results are repeatable)
   3. Decide    ask the two questions that best separate the leading conditions, or share the result
   4. Retrieve  find the relevant passages from NHS, MedlinePlus, ACOG and womenshealth.gov pages
   5. Reply     the AI writes a natural reply from those passages, with the whole conversation as context
```

Safety checks run on every turn: some combinations (sudden severe pain, a fever with a sunburn-like rash, possible pregnancy with pelvic pain and others) show an urgent-care message, and any mention of self-harm gets a reply with crisis lines.

## Files

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | The app: two sections, moon panel, chat |
| `rag.py` | Retrieval, AI calls, conversation memory, both agents |
| `assessment.py` | Symptom scoring and follow-up question choice |
| `knowledge/conditions.json` | 45 symptoms and 18 conditions, each linked to its source pages |
| `knowledge/sources.json`, `knowledge/web_sources.json` | Source text used for answers |
| `ingest.py`, `knowledge/urls.txt` | Rebuild the source library with `python ingest.py` |

## Limitations

Symptom weights are set by hand from the source pages and have not been clinically reviewed. A match percentage shows how well your answers fit a pattern, not the chance that you have the condition. Conditions outside the list (for example thyroid problems) can't be matched.
