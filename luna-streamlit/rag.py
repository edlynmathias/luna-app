"""Luna's RAG agent: semantic retrieval + a symptom-matching loop + conversation memory.

Pipeline per user turn (chat_turn):
  1. Understand the message in the context of the conversation (LLM extractor, keyword fallback):
     new present/absent symptoms, intent, and a standalone search query for follow-ups like "what about treatment?".
  2. Update the symptom state and score every condition (assessment.py, deterministic).
  3. Either ask the most informative follow-up questions, or present the closest matches.
  4. Retrieve source excerpts for the leading conditions (or the standalone query) and let the LLM write a
     cited reply, with the recent conversation passed in as chat history.

answer_question() is the original single-turn Q&A path; eval/ragas_eval.py scores it with RAGAS.
"""
import json, logging, os, re
from pathlib import Path

from assessment import Assessor, empty_state

log = logging.getLogger(__name__)

HISTORY_TURNS = 12          # messages of history sent to the LLM
MAX_MSG_CHARS = 4000

SYSTEM_PROMPT = (
    "You are Luna, a calm, warm menstrual-health education assistant. You are not a doctor and you never diagnose. "
    "You remember and use the earlier conversation: refer back to what the user already told you instead of asking again. "
    "Use only the evidence in the supplied excerpts for medical facts and cite them inline as [1], [2] matching the "
    "excerpt numbers. Never invent facts or citations. Write in plain, nonjudgmental language. Never use emojis."
)
PROMPT_TEMPLATE = (
    "Answer the exact question in age-appropriate, nonjudgmental language. Use ONLY evidence in the supplied excerpts; "
    "do not invent facts or citations. If evidence is insufficient, say so. Give proportionate urgent-care guidance "
    "when clearly warranted. Cite claims inline as [1], [2], matching the excerpt numbers. Keep the answer under 220 "
    "words; use short paragraphs or bullet points when listing symptoms or steps."
    "\n\nExcerpts:\n{context}\n\nQuestion: {question}"
)
NO_EVIDENCE = ("I couldn't find enough reliable information in my source library to answer that confidently. "
               "Try rephrasing, or ask a clinician or pharmacist for advice tailored to you.")
NOT_IN_SOURCES = ("I couldn't find this in my trusted health sources, so I can't give you a source-backed answer. "
                  "Would you like a general AI answer instead? It won't have citations, so double-check important "
                  "details with a clinician.")
GENERAL_LABEL = "Not found in Luna's trusted sources. This is a general AI answer, so double-check important details with a clinician.\n\n"
GENERAL_PROMPT = (
    "Luna's trusted source library had nothing on this question, so answer from well-established general medical "
    "knowledge only. Be accurate and cautious: do not give drug doses or diagnoses, say when something is uncertain, "
    "and suggest seeing a clinician when appropriate. If the question is unrelated to health, periods or the body, "
    "politely say you can only help with menstrual and reproductive health. Do not cite sources. "
    "Keep the answer under 180 words.\n\nQuestion: {question}"
)
CRISIS_REPLY = (
    "I'm really glad you told me. If you're having thoughts of harming yourself or ending your life, please reach out "
    "for support right now. You can call your local emergency number, or a crisis line: in India, Tele-MANAS on 14416; "
    "in the UK, Samaritans on 116 123; in the US, call or text 988. If you can, tell someone you trust how you're feeling. "
    "Severe low mood before periods can be part of PMDD, and it is treatable, so a doctor is a good next step when you feel able. "
    "I'm here if you'd like to keep talking."
)
SMALL_TALK = ("I can answer questions about periods and menstrual health, or you can describe what you're "
              "experiencing and I'll ask a few questions to see which condition it matches most closely.")
DISCLAIMER = "This is a match against common symptom patterns, not a diagnosis. Only a clinician can confirm what's going on."


# ============================================================== retrieval

def get_embedder():
    """all-MiniLM-L6-v2 via ONNX Runtime (bundled with chromadb) - no PyTorch needed."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return lambda texts: [[float(x) for x in v] for v in ef(list(texts))]


def chunk_text(text, max_words=120, overlap_sentences=1):
    """Sentence-aware chunking with a one-sentence overlap. Short docs stay a single chunk."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class KnowledgeBase:
    def __init__(self, min_score=None):
        self.docs, self.chunks = [], []
        self.collection = self.embedder = None
        self.mode = "keyword fallback"
        self.min_score = float(min_score if min_score is not None else os.getenv("MIN_RELEVANCE", "0.30"))

    def ingest(self, registry_path):
        """Load curated sources.json plus web_sources.json (from ingest.py) if present, chunk and index them."""
        registry_path = Path(registry_path)
        self.docs = json.loads(registry_path.read_text(encoding="utf-8"))
        web = registry_path.with_name("web_sources.json")
        if web.exists():
            self.docs += json.loads(web.read_text(encoding="utf-8"))
        self.chunks = [
            {"title": d["title"], "url": d["url"], "doc_id": d["id"], "id": f"{d['id']}#{i}", "text": c,
             "keywords": d.get("keywords", [])}
            for d in self.docs for i, c in enumerate(chunk_text(d["text"]))
        ]
        try:
            import chromadb, hashlib
            self.embedder = get_embedder()
            client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", ".chroma"))
            fingerprint = hashlib.sha1(json.dumps([(c["id"], c["text"]) for c in self.chunks]).encode()).hexdigest()
            try:
                col = client.get_collection("luna_health_sources")
                reuse = (col.metadata or {}).get("fingerprint") == fingerprint and col.count() == len(self.chunks)
            except Exception:
                col, reuse = None, False
            if reuse:
                self.collection = col
            else:
                if col is not None:
                    client.delete_collection("luna_health_sources")
                self.collection = client.create_collection(
                    "luna_health_sources", metadata={"hnsw:space": "cosine", "fingerprint": fingerprint})
                for b in range(0, len(self.chunks), 256):
                    batch = self.chunks[b:b + 256]
                    embs = self.embedder([f"{c['title']}: {c['text']}" for c in batch])
                    self.collection.add(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                                        embeddings=embs,
                                        metadatas=[{"doc_id": c["doc_id"], "title": c["title"], "url": c["url"]} for c in batch])
            self.mode = "semantic embeddings + Chroma"
        except Exception as e:
            log.warning("Semantic index unavailable, using keyword fallback: %s", e)
            self.collection = self.embedder = None
            self.mode = "keyword fallback"
        return {"documents": len(self.docs), "chunks": len(self.chunks), "mode": self.mode}

    def retrieve(self, query, k=5, urls=None, min_score=None):
        """Return up to k chunks (each with a `score`). `urls` restricts results to those source pages."""
        floor = self.min_score if min_score is None else min_score
        pool = [c for c in self.chunks if not urls or c["url"] in urls]
        if not pool:
            return []
        if self.collection is not None:
            kwargs = {"where": {"url": {"$in": list(urls)}}} if urls else {}
            hits = self.collection.query(query_embeddings=self.embedder([query]), n_results=min(k, len(pool)), **kwargs)
            by_id = {c["id"]: c for c in self.chunks}
            out = []
            for cid, dist in zip(hits["ids"][0], hits["distances"][0]):
                score = 1 - dist
                if cid in by_id and score >= floor:
                    out.append({**by_id[cid], "score": round(score, 4)})
            return out
        terms = set(re.findall(r"[a-z]{3,}", query.lower()))
        scored = []
        for c in pool:
            text = (c["title"] + " " + c["text"] + " " + " ".join(c["keywords"])).lower()
            hit = sum(1 for t in terms if re.search(r"\b" + re.escape(t), text))
            if hit:
                scored.append({**c, "score": round(hit / max(len(terms), 1), 4)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]


def _numbering(chunks):
    """One citation number per unique source, so [n] in the answer matches the source list."""
    nums = {}
    for c in chunks:
        nums.setdefault(c["url"], len(nums) + 1)
    return nums


def _context_block(chunks):
    nums = _numbering(chunks)
    return "\n".join(f"[{nums[c['url']]}] {c['title']} ({c['url']}): {c['text']}" for c in chunks)


def build_prompt(question, chunks):
    return PROMPT_TEMPLATE.format(context=_context_block(chunks), question=question)


def _sources(chunks):
    titles = {c["url"]: c["title"] for c in chunks}
    return [{"n": n, "title": titles[u], "url": u} for u, n in _numbering(chunks).items()]


# ==================================================================== LLM

def llm_configured():
    key = os.getenv("LLM_API_KEY", "")
    return bool(key) and key != "replace_me"


def chat(messages, json_mode=False, retries=5, temperature=None):
    """Call an OpenAI-compatible chat-completions API with retries on 429 / temporary 5xx."""
    import time, requests
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {"model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")) if temperature is None else temperature,
            "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(retries + 1):
        r = requests.post(base + "/chat/completions", json=body, timeout=40,
                          headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"})
        if r.status_code == 400 and json_mode and "response_format" in body:
            body.pop("response_format")          # some providers don't support JSON mode
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < retries:
            wait = float(r.headers.get("Retry-After") or 0) or min(2 ** attempt, 20)
            log.warning("LLM API %s, retrying in %.0fs", r.status_code, wait)
            time.sleep(wait)
            continue
        if not r.ok:
            raise RuntimeError(f"{r.status_code} from {base}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("LLM API kept failing after retries")


def generate(prompt):
    """Single-prompt helper kept for the original Q&A path and RAGAS evaluation."""
    return chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])


def general_answer(question, history=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + clean_history(history)
    msgs.append({"role": "user", "content": GENERAL_PROMPT.format(question=question)})
    return GENERAL_LABEL + chat(msgs)


def answer_question(question, kb, k=5):
    """Original single-turn grounded Q&A (no memory). Used by tests and eval/ragas_eval.py."""
    chunks = kb.retrieve(question, k=k)
    mode = getattr(kb, "mode", "unknown")
    if not chunks:
        policy = os.getenv("GENERAL_ANSWERS", "ask").lower()
        can = llm_configured() and policy != "off"
        if can and policy == "auto":
            return {"answer": general_answer(question), "sources": [], "contexts": [], "retrieval_mode": mode,
                    "grounded": False, "offer_general": False}
        return {"answer": NOT_IN_SOURCES if can else NO_EVIDENCE, "sources": [], "contexts": [],
                "retrieval_mode": mode, "grounded": False, "offer_general": can}
    answer = generate(build_prompt(question, chunks)) if llm_configured() else _excerpt_summary(chunks)
    return {"answer": answer, "sources": _sources(chunks), "contexts": [c["text"] for c in chunks],
            "retrieval_mode": mode, "grounded": True, "offer_general": False}


def _excerpt_summary(chunks, limit=900):
    seen, sents = set(), []
    for c in chunks[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            if sent not in seen and sum(len(x) for x in sents) + len(sent) < limit:
                seen.add(sent); sents.append(sent)
    return ("No AI provider is configured, so this is a summary of the source excerpts rather than a generated answer.\n\n"
            + " ".join(sents))


# ================================================================ memory

def clean_history(history):
    """Keep the last HISTORY_TURNS valid {role, content} messages."""
    out = []
    for m in (history or [])[-HISTORY_TURNS:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"][:MAX_MSG_CHARS]})
    return out


def _history_text(history, n=6):
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in clean_history(history)[-n:])


def standalone_query_fallback(message, history):
    """Without an LLM, make short follow-ups ("what about treatment?") searchable by adding the previous user turn."""
    prev = [m["content"] for m in clean_history(history) if m["role"] == "user"]
    if prev and (len(message.split()) <= 7 or re.search(r"\b(it|this|that|they|them|those|these)\b", message, re.I)):
        return prev[-1] + " " + message
    return message


EXTRACT_PROMPT = """You read a conversation between a user and Luna, a menstrual-health assistant, and return JSON only.

Symptom vocabulary (id: meaning):
{vocab}

Questions Luna just asked (ids, in order): {pending}

Recent conversation:
{history}

New user message: {message}

Return exactly this JSON object:
{{"present": [ids the user now says they HAVE], "absent": [ids the user now says they do NOT have],
  "intent": "assess" | "question" | "results" | "crisis" | "other",
  "standalone_query": "the user's request rewritten as a self-contained search query using the conversation"}}

Rules: only use ids from the vocabulary. Map short replies ("yes", "no to the second", "the first two") onto the
questions Luna just asked. intent is "assess" when the user describes their own symptoms or answers Luna's questions,
"results" when they ask what condition matches or want the outcome, "question" for general information questions,
"crisis" if they mention self-harm or suicide, otherwise "other"."""


def understand(message, history, state, assessor):
    """Extract symptoms + intent from the message using conversation context. Falls back to rules."""
    kw_present, kw_absent = assessor.extract_keywords(message, state["pending"])
    fallback = {"present": kw_present, "absent": kw_absent,
                "intent": assessor.detect_intent(message, state, kw_present),
                "standalone_query": standalone_query_fallback(message, history), "extractor": "rules"}
    if not llm_configured():
        return fallback
    try:
        raw = chat([{"role": "user", "content": EXTRACT_PROMPT.format(
            vocab=assessor.vocabulary(), pending=state["pending"] or "none",
            history=_history_text(history) or "(start of conversation)", message=message)}],
            json_mode=True, temperature=0)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        valid = lambda xs: [x for x in (xs or []) if x in assessor.symptoms]
        intent = data.get("intent") if data.get("intent") in ("assess", "question", "results", "crisis", "other") else fallback["intent"]
        if fallback["intent"] == "crisis":
            intent = "crisis"
        return {"present": valid(data.get("present")), "absent": valid(data.get("absent")), "intent": intent,
                "standalone_query": str(data.get("standalone_query") or fallback["standalone_query"])[:500],
                "extractor": "llm"}
    except Exception as e:
        log.warning("LLM extraction failed, using rules: %s", e)
        return fallback


# ================================================================== agent

def _retrieve_for_matches(kb, matches, state, assessor, per=2, top=3):
    labels = ", ".join(assessor.symptoms[s]["label"] for s in state["present"])
    chunks, seen = [], set()
    for m in matches[:top]:
        for c in kb.retrieve(f"{m['name']} symptoms: {labels}", k=per, urls=m["sources"], min_score=0.0):
            if c["id"] not in seen:
                seen.add(c["id"]); chunks.append(c)
    return chunks


def _format_matches(matches, n=4):
    lines = []
    for m in matches[:n]:
        line = f"- {m['name']}: {m['percent']}% match. Fits: {', '.join(m['matched']) or 'none'}."
        if m["against"]:
            line += f" Doesn't fit: {', '.join(m['against'])}."
        if m["unexplained"]:
            line += f" Not explained by it: {', '.join(m['unexplained'])}."
        lines.append(line)
    return "\n".join(lines) or "- No clear match yet."


def _local_assess_reply(state, matches, questions, ready, assessor):
    noted = ", ".join(assessor.symptoms[s]["label"].lower() for s in state["present"]) or "nothing specific yet"
    if ready and matches:
        top = matches[0]
        txt = (f"Based on everything you've told me ({noted}), the closest match is {top['name']} "
               f"at {top['percent']}%. {top['summary']}")
        if len(matches) > 1:
            txt += " Other possibilities: " + ", ".join(f"{m['name']} ({m['percent']}%)" for m in matches[1:3]) + "."
        return txt + "\n\n" + DISCLAIMER
    lead = f" So far it looks most like {matches[0]['name']}, but I need a little more to be sure." if matches else ""
    return f"Thanks. So far I've noted: {noted}.{lead} A few questions will help me narrow it down."


def chat_turn(message, history, state, kb, assessor, answers=None, force_general=False):
    """One conversational turn. Returns the reply plus the updated state for the client to send back next time."""
    state = assessor.sanitize(state)
    history = clean_history(history)
    mode_name = getattr(kb, "mode", "unknown")
    base = {"sources": [], "contexts": [], "retrieval_mode": mode_name, "grounded": False, "offer_general": False}

    if force_general and llm_configured():
        text = general_answer(message, history)
        return {**base, "answer": text, "history_text": text, "state": state, "assessment": None, "intent": "question"}

    # 1. understand
    if answers:
        yes = [k for k, v in answers.items() if v is True]
        no = [k for k, v in answers.items() if v is False]
        info = {"present": yes, "absent": no, "intent": "assess", "standalone_query": message, "extractor": "buttons"}
    else:
        info = understand(message, history, state, assessor)

    if info["intent"] == "crisis":
        return {**base, "answer": CRISIS_REPLY, "history_text": CRISIS_REPLY, "state": state,
                "assessment": None, "intent": "crisis"}

    # 2. update state
    assessor.apply(state, info["present"], info["absent"])
    state["asked"] = list(dict.fromkeys(state["asked"] + state["pending"]))
    state["pending"] = []
    intent = info["intent"]
    if intent in ("assess", "results") or (state["mode"] == "assess" and intent == "other" and (info["present"] or info["absent"])):
        state["mode"], intent = "assess", ("results" if intent == "results" else "assess")

    # 3a. small talk, or a general question -> grounded Q&A with memory
    if intent == "other" and state["mode"] == "qa":
        text = SMALL_TALK
        return {**base, "answer": text, "history_text": text, "state": state, "assessment": None, "intent": "other"}
    if intent not in ("assess", "results"):
        if state["mode"] == "assess" and not QUESTIONISH(message):
            intent = "assess"
        else:
            query = info["standalone_query"]
            if info["extractor"] == "rules" and state["present"] and PRONOUN.search(message):
                ranked = assessor.score(state)          # "treatment for it?" -> the leading condition
                if ranked:
                    query = f"{ranked[0]['name']} {message}"
            return _qa_turn(message, query, history, state, kb, base)

    # 3b. symptom assessment
    matches = assessor.score(state)
    questions = assessor.next_questions(state, matches)
    ready = assessor.is_ready(state, matches, questions, intent)
    urgent = assessor.urgent(state)
    if not ready:
        state["pending"] = questions
        state["rounds"] += 1
    chunks = _retrieve_for_matches(kb, matches, state, assessor) if matches else []
    q_payload = assessor.questions_payload(questions) if not ready else []

    if llm_configured():
        if ready:
            task = ("Present the result now. Name the closest match and its percentage first, then up to two runner-ups. "
                    "For each, say in a sentence which of the user's symptoms fit and which don't, citing the excerpts. "
                    "Then say what a clinician might check or ask about, and suggest booking a GP or gynaecologist visit. "
                    f"End with this sentence exactly: \"{DISCLAIMER}\" Keep it under 260 words. Do not change the ranking.")
        else:
            task = ("Do NOT give a final answer yet. In two or three sentences, reflect back what you've understood "
                    "so far, referring to details from earlier in the conversation. If the picture already leans "
                    "somewhere, mention it tentatively. Then say you have a few questions to narrow it down. "
                    "Do NOT write the questions yourself; the app shows them below your message. Under 90 words.")
        context = (f"[Information for Luna, not shown to the user]\n"
                   f"Symptoms the user has: {', '.join(assessor.symptoms[s]['label'] for s in state['present']) or 'none yet'}\n"
                   f"Symptoms the user does not have: {', '.join(assessor.symptoms[s]['label'] for s in state['absent']) or 'none'}\n"
                   f"Ranked matches from Luna's matching engine:\n{_format_matches(matches)}\n"
                   f"Excerpts:\n{_context_block(chunks) or '(none)'}\n\nTask: {task}")
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
            {"role": "user", "content": f"{message}\n\n{context}"}]
        answer = chat(msgs)
    else:
        answer = _local_assess_reply(state, matches, questions, ready, assessor)

    if urgent:
        answer = "Please act on this first: " + " ".join(urgent) + "\n\n" + answer
    history_text = answer
    if q_payload:
        history_text += "\n\nQuestions:\n" + "\n".join(f"{i}. {q['text']}" for i, q in enumerate(q_payload, 1))

    return {**base, "answer": answer, "history_text": history_text, "sources": _sources(chunks),
            "contexts": [c["text"] for c in chunks], "grounded": bool(chunks), "state": state, "intent": intent,
            "assessment": {"matches": matches[:5], "questions": q_payload, "ready": ready, "urgent": urgent,
                           "symptoms": assessor.picture(state), "rounds": state["rounds"],
                           "completeness": assessor.completeness(state, matches), "extractor": info["extractor"]}}


PRONOUN = re.compile(r"\b(it|this|that|they|them|those|these)\b", re.I)


def QUESTIONISH(text):
    return bool(re.search(r"\?\s*$", text.strip())) or bool(re.match(
        r"^\s*(what|how|why|when|where|which|who|can|could|should|is|are|do|does|will|tell me about)\b", text, re.I))


def _qa_turn(message, query, history, state, kb, base):
    chunks = kb.retrieve(query, k=5)
    if not chunks:
        policy = os.getenv("GENERAL_ANSWERS", "ask").lower()
        can = llm_configured() and policy != "off"
        if can and policy == "auto":
            text = general_answer(message, history)
            return {**base, "answer": text, "history_text": text, "state": state, "assessment": None, "intent": "question"}
        text = NOT_IN_SOURCES if can else NO_EVIDENCE
        return {**base, "answer": text, "history_text": text, "offer_general": can, "state": state,
                "assessment": None, "intent": "question"}
    if llm_configured():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": (
            f"{message}\n\n[Information for Luna, not shown to the user]\nSearch query used: {query}\n"
            f"Excerpts:\n{_context_block(chunks)}\n\nTask: answer the user's latest message using the conversation for "
            "context and only these excerpts for facts. Cite inline as [n]. Under 220 words. If they describe "
            "their own symptoms, offer to go through a few questions to see which condition matches best.")}]
        text = chat(msgs)
    else:
        text = _excerpt_summary(chunks)
    return {**base, "answer": text, "history_text": text, "sources": _sources(chunks),
            "contexts": [c["text"] for c in chunks], "grounded": True, "state": state, "assessment": None,
            "intent": "question"}


__all__ = ["KnowledgeBase", "Assessor", "empty_state", "chat_turn", "answer_question", "general_answer", "chunk_text"]
