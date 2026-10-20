# Luna: menstrual-health RAG agent

Describe your symptoms and Luna asks follow-up questions, remembers the conversation, and shows which condition your symptoms match most closely, with citations from NHS, MedlinePlus and ACOG pages. Educational only, not a diagnosis.

## Deploy on Streamlit Community Cloud

1. Push this folder to a new GitHub repository.
2. Go to https://share.streamlit.io, choose **Create app**, and pick your repository and branch.
3. Set **Main file path** to `streamlit_app.py`.
4. Open **Advanced settings**, choose **Python 3.12**, and paste into **Secrets**:
   ```
   LLM_API_KEY = "gsk_your_key_here"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
   A free Groq key is available at https://console.groq.com/keys. Any OpenAI-compatible provider works.
5. Click **Deploy**. The first start takes a few minutes while it installs packages and downloads the small embedding model.

## Run locally

```
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your key
streamlit run streamlit_app.py
```

## Files

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | The app |
| `rag.py` | Retrieval, LLM calls, conversation memory, agent loop |
| `assessment.py` | Symptom scoring and follow-up question selection |
| `knowledge/conditions.json` | Symptoms and condition profiles with sources |
| `knowledge/*sources.json` | Source text used for retrieval |
| `ingest.py`, `knowledge/urls.txt` | Rebuild the source library: `python ingest.py` |
