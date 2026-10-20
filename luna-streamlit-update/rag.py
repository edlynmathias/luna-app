"""Luna's agents: grounded retrieval + conversation memory, in two modes.

symptom_turn(): the symptom check. Luna reads each message in the context of the whole conversation, updates the
    symptom picture, decides whether to ask more (two questions at most, woven into a natural reply) or to share
    which condition fits best, and answers side questions without losing her place.
qa_turn(): general questions about periods ("how do I ease cramps?"), answered from trusted sources with citations.
    Follow-ups like "what about at night?" are rewritten into a full search query using the conversation.

The matching itself (assessment.py) is deterministic; the LLM only reads messages and writes replies.
"""
import json, logging, os, random, re
from pathlib import Path

from assessment import Assessor, empty_state

log = logging.getLogger(__name__)

HISTORY_TURNS = 14
MAX_MSG_CHARS = 4000

SYSTEM_PROMPT = (
    "You are Luna, a warm, calm guide for periods and menstrual health. You talk like a kind, knowledgeable friend: "
    "natural sentences, contractions, plain words, no corporate tone, no headings, never any emojis. You remember the "
    "whole conversation and refer back to specific things the person told you instead of asking again. You never sound "
    "like a form or a checklist: don't repeat their symptoms back as a list, and don't start replies with 'Thanks for "
    "sharing' or 'Based on the information provided'. You are not a doctor and never claim to diagnose. For medical "
    "facts, use only the excerpts you are given and cite them inline as [1], [2]. If the excerpts don't cover "
    "something, say so plainly rather than guessing."
)
HIDDEN = "\n\n(Notes for Luna only. The person cannot see these notes; never mention them, the matching engine or percentages unless told to.)\n"

CRISIS_REPLY = (
    "I'm really glad you told me, and I'm sorry you're feeling this way. If you're having thoughts of harming yourself "
    "or ending your life, please reach out for support right now. In India you can call Tele-MANAS on 14416, in the UK "
    "Samaritans on 116 123, and in the US you can call or text 988. Your local emergency number works too. If you can, "
    "let someone you trust know how you're feeling.\n\nVery low mood before periods can be part of PMDD, and it is "
    "treatable, so talking to a doctor is a good next step when you feel able. I'm here if you want to keep talking."
)
NO_EVIDENCE = ("I couldn't find anything reliable on that in the health sources I use, so I'd rather not guess. "
               "Could you put it another way? A doctor or pharmacist can also help with something this specific.")
OFFER_GENERAL = ("I couldn't find that in the health sources I rely on. I can still give you a general answer from what I "
                 "know, just without citations, if that would help. Want me to?")
GENERAL_PREFIX = "This one isn't covered by my trusted sources, so it's from general knowledge and worth double-checking with a clinician.\n\n"


# ============================================================== retrieval

def get_embedder():
    """all-MiniLM-L6-v2 via ONNX Runtime (bundled with chromadb), no PyTorch needed."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return lambda texts: [[float(x) for x in v] for v in ef(list(texts))]


def chunk_text(text, max_words=120, overlap_sentences=1):
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
            client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", str(registry_path.parent.parent / ".chroma")))
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
        floor = self.min_score if min_score is None else min_score
        pool = [c for c in self.chunks if not urls or c["url"] in urls]
        if not pool:
            return []
        if self.collection is not None:
            kwargs = {"where": {"url": {"$in": list(urls)}}} if urls else {}
            hits = self.collection.query(query_embeddings=self.embedder([query]), n_results=min(k, len(pool)), **kwargs)
            by_id = {c["id"]: c for c in self.chunks}
            return [{**by_id[cid], "score": round(1 - dist, 4)}
                    for cid, dist in zip(hits["ids"][0], hits["distances"][0])
                    if cid in by_id and 1 - dist >= floor]
        terms = set(re.findall(r"[a-z]{3,}", query.lower())) - STOPWORDS
        if not terms:
            return []
        idf = {t: self._idf(t) for t in terms}
        total = sum(idf.values())
        scored = []
        for c in pool:
            title, body = c["title"].lower(), (c["text"] + " " + " ".join(c["keywords"])).lower()
            got = 0.0
            for t in terms:
                pat = r"\b" + re.escape(t[:-1] if t.endswith("s") and len(t) > 4 else t)
                if re.search(pat, title): got += idf[t] * 1.3
                elif re.search(pat, body): got += idf[t]
            if got:
                scored.append({**c, "score": round(min(got / total, 1.0), 4)})
        scored = [c for c in scored if c["score"] >= (0.0 if urls else 0.45)]
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]

    def _idf(self, term):
        if not hasattr(self, "_df"):
            self._df, self._n = {}, max(len(self.chunks), 1)
            for c in self.chunks:
                for w in set(re.findall(r"[a-z]{3,}", (c["title"] + " " + c["text"]).lower())):
                    self._df[w] = self._df.get(w, 0) + 1
        import math
        stem = term[:-1] if term.endswith("s") and len(term) > 4 else term
        df = max(self._df.get(term, 0), self._df.get(stem, 0))
        return math.log((self._n + 1) / (df + 1)) + 0.1


STOPWORDS = {"the", "and", "how", "what", "can", "for", "with", "you", "does", "about", "should", "are", "is", "my",
             "get", "have", "has", "had", "this", "that", "when", "why", "which", "who", "there", "from", "your", "any",
             "will", "would", "could", "much", "many", "more", "most", "some", "just", "really", "like", "also", "into",
             "during", "ease", "help", "helps", "way", "ways", "things", "thing", "need", "know", "tell", "luna", "them",
             "they", "their", "too", "very", "ok", "okay", "it's", "its", "not", "but", "use", "using", "eat", "do"}


def _numbering(chunks):
    nums = {}
    for c in chunks:
        nums.setdefault(c["url"], len(nums) + 1)
    return nums


def _context_block(chunks):
    nums = _numbering(chunks)
    return "\n".join(f"[{nums[c['url']]}] {c['title']}: {c['text']}" for c in chunks)


def _sources(chunks):
    titles = {c["url"]: c["title"] for c in chunks}
    return [{"n": n, "title": titles[u], "url": u} for u, n in _numbering(chunks).items()]


# ==================================================================== LLM

def llm_configured():
    key = os.getenv("LLM_API_KEY", "")
    return bool(key) and key not in ("replace_me", "gsk_your_key_here")


def chat(messages, json_mode=False, retries=4, temperature=None):
    """Call an OpenAI-compatible chat-completions API, retrying on 429 / temporary 5xx."""
    import time, requests
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    body = {"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.4")) if temperature is None else temperature,
            "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(retries + 1):
        r = requests.post(base + "/chat/completions", json=body, timeout=45,
                          headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"})
        if r.status_code == 400 and json_mode and "response_format" in body:
            body.pop("response_format")
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < retries:
            wait = float(r.headers.get("Retry-After") or 0) or min(2 ** attempt, 20)
            time.sleep(wait)
            continue
        if not r.ok:
            raise RuntimeError(f"{r.status_code} from {base}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("The AI provider kept failing after several retries.")


def write(history, message, notes):
    """Luna's voice: system persona + conversation so far + the new message with private notes."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + clean_history(history)
    msgs.append({"role": "user", "content": message + HIDDEN + notes})
    return _strip_emoji(chat(msgs))


_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
def _strip_emoji(t):
    return _EMOJI.sub("", t).strip()


def clean_history(history):
    out = []
    for m in (history or [])[-HISTORY_TURNS:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"][:MAX_MSG_CHARS]})
    return out


def _history_text(history, n=8):
    return "\n".join(f"{'PERSON' if m['role'] == 'user' else 'LUNA'}: {m['content']}" for m in clean_history(history)[-n:])


PRONOUN = re.compile(r"\b(it|this|that|they|them|those|these|its)\b", re.I)


def rewrite_query(message, history, extra=""):
    """Turn a follow-up into a self-contained search query using the conversation."""
    hist = clean_history(history)
    if not hist:
        return message
    needs = len(message.split()) <= 10 or PRONOUN.search(message)
    if not needs:
        return message
    if llm_configured():
        try:
            q = chat([{"role": "user", "content": (
                "Rewrite the person's latest message as one self-contained search query about menstrual health, "
                "resolving words like 'it' or 'that' from the conversation. Reply with the query only.\n\n"
                f"Conversation:\n{_history_text(hist, 6)}\n\nLatest message: {message}{extra}")}], temperature=0)
            return q.strip().strip('"')[:300] or message
        except Exception as e:
            log.warning("Query rewrite failed: %s", e)
    prev = [m["content"] for m in hist if m["role"] == "user"]
    follow_up = PRONOUN.search(message) or re.match(r"^\s*(what about|and|how about|also)\b", message, re.I)
    return (prev[-1] + " " + message) if prev and follow_up else message


def _excerpt_summary(chunks, limit=850):
    seen, sents = set(), []
    for c in chunks[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            if sent not in seen and sum(len(x) for x in sents) + len(sent) < limit:
                seen.add(sent); sents.append(sent)
    return "Here's what my sources say about that:\n\n" + " ".join(sents)


YES_SHORT = re.compile(r"^\s*(yes|yeah|yep|yup|ya|sure|ok(ay)?|please|go on|go ahead|do it)\b", re.I)


# ========================================================= general Q&A

def qa_turn(message, history, state, kb):
    """General period questions. state = {"awaiting_general": question or None}."""
    state = dict(state or {})
    pending_general = state.pop("awaiting_general", None)
    base = {"sources": [], "urgent": [], "assessment": None, "state": {"awaiting_general": None}}

    if pending_general and YES_SHORT.search(message) and len(message.split()) <= 6 and llm_configured():
        text = GENERAL_PREFIX + write(history, message, (
            f"Earlier they asked: {pending_general}\nNothing in your trusted sources covered it, and they've just agreed "
            "to a general answer. Answer that question from well-established general medical knowledge. Be accurate and "
            "cautious: no drug doses, no diagnoses, say when something is uncertain, and suggest a clinician when it "
            "makes sense. No citations. Under 170 words."))
        return {**base, "answer": text}

    if CRISIS(message):
        return {**base, "answer": CRISIS_REPLY}

    query = rewrite_query(message, history)
    chunks = kb.retrieve(query, k=5)
    if not chunks:
        if llm_configured():
            return {**base, "answer": OFFER_GENERAL, "state": {"awaiting_general": query}}
        return {**base, "answer": NO_EVIDENCE}
    if llm_configured():
        text = write(history, message, (
            f"Search query used: {query}\nExcerpts:\n{_context_block(chunks)}\n\n"
            "Answer the person's latest message, using the conversation for context and only these excerpts for facts. "
            "Explain it the way a friend who knows the NHS pages would: short paragraphs, and a short list only if you "
            "are giving practical steps. Cite inline as [n]. Under 200 words. Say when something needs a doctor. If they "
            "describe their own symptoms and want to know what might be causing them, mention that the Symptom check "
            "section can go through it with them."))
    else:
        text = _excerpt_summary(chunks)
    return {**base, "answer": text, "sources": _sources(chunks)}


def GREET(text):
    from assessment import GREETING
    return bool(GREETING.search(text))


def CRISIS(text):
    from assessment import CRISIS as C
    return bool(C.search(text))


# ======================================================== symptom check

EXTRACT_PROMPT = """You read a conversation between a person and Luna, a menstrual-health guide, and return JSON only.

Symptom vocabulary (id: meaning):
{vocab}

Questions Luna asked in her last message (in order): {pending}

Conversation so far:
{history}

Person's new message: {message}

Return exactly this JSON object:
{{"present": [ids the person now says they HAVE],
  "absent": [ids the person now says they do NOT have],
  "intent": "assess" | "question" | "results" | "crisis" | "other"}}

Rules: only use ids from the vocabulary. Map short replies ("yes", "no to the second", "the first one, sometimes")
onto the questions Luna just asked, in order. "Sometimes" or "a little" counts as having it. Include symptoms mentioned
anywhere in the new message. intent: "assess" when they describe their own symptoms or answer Luna's questions;
"question" when they ask for information (for example how to ease pain or what a condition is) without new symptoms;
"results" when they ask what condition it is or want Luna's conclusion; "crisis" if they mention self-harm or suicide;
otherwise "other"."""


def understand(message, history, state, assessor):
    kw_p, kw_a = assessor.extract_keywords(message, state["pending"])
    fallback = {"present": kw_p, "absent": kw_a, "intent": assessor.detect_intent(message, state, kw_p or kw_a),
                "extractor": "rules"}
    if not llm_configured():
        return fallback
    try:
        pend = "; ".join(f"{i}. {s} ({assessor.symptoms[s]['question']})" for i, s in enumerate(state["pending"], 1)) or "none"
        raw = chat([{"role": "user", "content": EXTRACT_PROMPT.format(
            vocab=assessor.vocabulary(), pending=pend, history=_history_text(history) or "(start)", message=message)}],
            json_mode=True, temperature=0)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        valid = lambda xs: [x for x in (xs or []) if x in assessor.symptoms]
        intent = data.get("intent") if data.get("intent") in ("assess", "question", "results", "crisis", "other") else fallback["intent"]
        if fallback["intent"] == "crisis":
            intent = "crisis"
        return {"present": valid(data.get("present")), "absent": valid(data.get("absent")), "intent": intent, "extractor": "llm"}
    except Exception as e:
        log.warning("LLM extraction failed, using rules: %s", e)
        return fallback


def _natural_list(items):
    items = [i[0].lower() + i[1:] for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _ask_sentence(qtexts):
    q = [t[0].upper() + t[1:] for t in qtexts]
    if len(q) == 1:
        return q[0]
    return q[0] + " And " + qtexts[1]


def _fallback_reply(kind, state, matches, qtexts, rng, first):
    if kind == "open":
        return ("Of course. Tell me what's been going on, in your own words. Anything helps: when it happens in your "
                "cycle, any pain, how heavy your periods are, or changes in your mood or skin.")
    if kind == "ask":
        ack = rng.choice(["That sounds really tough to deal with.", "I'm sorry, that sounds exhausting.",
                          "Thank you for telling me, that helps."] if first else
                         ["Okay, that helps.", "Thanks, that's useful to know.", "Got it, thank you.", "Okay, that makes sense."])
        lean = ""
        if matches and not first and matches[0]["score"] >= 0.45:
            lean = " " + rng.choice([f"It's starting to sound like it could be {matches[0]['short']}, but I want to check a little more.",
                                     f"I have a feeling this might be {matches[0]['short']}, though I'd like to be surer."])
        lead = rng.choice(["Can I ask you a couple of things?", "Could you tell me a bit more?", "A couple of questions, if that's okay."])
        return f"{ack}{lean} {lead} {_ask_sentence(qtexts)}"
    top = matches[0]
    txt = (f"Thank you for bearing with all my questions. Putting together everything you've told me, the pattern that fits "
           f"best is {top['name']}. {top['summary']}")
    if top["matched"]:
        txt += f" It lines up with your {_natural_list(top['matched'])}."
    if top["against"]:
        txt += f" The part that doesn't quite fit is {_natural_list(top['against'])}."
    rest = [m["short"] for m in matches[1:3] if m["percent"] >= 25]
    if rest:
        txt += f"\n\nIt could also be {' or '.join(rest)}, so {'that is' if len(rest) == 1 else 'those are'} worth keeping in mind."
    return txt + ("\n\nThis isn't a diagnosis, just how your symptoms compare with common patterns. A GP or gynaecologist "
                  "can look into it properly, and it really helps to take a note of your symptoms and when in your cycle "
                  "they happen. If you'd like, ask me anything about it, like what usually helps.")


def symptom_turn(message, history, state, kb, assessor):
    """One turn of the symptom check. Returns the reply and the updated state."""
    state = assessor.sanitize(state)
    history = clean_history(history)
    rng = random.Random(message + str(state["rounds"]))
    out = {"sources": [], "urgent": [], "assessment": None}

    info = understand(message, history, state, assessor)
    if info["intent"] == "crisis":
        return {**out, "answer": CRISIS_REPLY, "state": state}
    found = bool(info["present"] or info["absent"])

    # A side question ("how can I ease the pain?"): answer it, keep the open questions for later.
    if info["intent"] == "question" and not found:
        matches = assessor.score(state)
        lead = f" (they are going through symptoms that currently fit {matches[0]['name']} best)" if matches else ""
        if llm_configured():
            query = rewrite_query(message, history, extra=lead)
        else:
            query = f"{matches[0]['name']} {message}" if matches else message
        chunks = kb.retrieve(query, k=5) or (kb.retrieve(message, k=5) if query != message else [])
        pending_q = [assessor.symptoms[s]["question"] for s in state["pending"]]
        if llm_configured():
            back = (f" Then, in one short natural sentence, gently come back to what you were asking: {' / '.join(pending_q)}."
                    if pending_q else "")
            if chunks:
                text = write(history, message, (
                    f"Excerpts:\n{_context_block(chunks)}\n\nAnswer their question warmly from these excerpts, citing [n], "
                    f"in under 170 words.{back}"))
            else:
                text = write(history, message, (
                    "Your trusted sources had nothing on this. Say so honestly in a sentence and suggest a doctor or "
                    f"pharmacist could help.{back}"))
        else:
            text = _excerpt_summary(chunks) if chunks else NO_EVIDENCE
            if pending_q:
                text += f"\n\nWhenever you're ready, I'd still like to know: {pending_q[0]}"
        return {**out, "answer": text, "sources": _sources(chunks), "state": state,
                "assessment": _assessment(state, assessor.score(state), [], False, assessor)}

    # Already shared a result and nothing new: a short, human follow-up instead of repeating it.
    if state["concluded"] and not found:
        matches = assessor.score(state)
        top = matches[0] if matches else None
        if llm_configured():
            task = (f"You already shared that their symptoms fit {top['name']} best. They asked again; remind them in two "
                    "or three sentences, without repeating your earlier answer word for word, and offer to explain what "
                    "helps or what a doctor might do." if info["intent"] == "results" and top else
                    "You've already shared your thoughts. Reply briefly and naturally to their message. Let them know they "
                    "can tell you about any new symptoms and you'll factor them in, or ask anything about it. Under 50 words.")
            text = write(history, message, task)
        elif info["intent"] == "results" and top:
            text = (f"From everything you've told me, it still fits {top['short']} best. If you'd like, I can tell you more "
                    "about it, or what usually helps. And if you notice anything new, just tell me and I'll factor it in.")
        else:
            text = ("Okay. If you notice anything else, like a new symptom or something changing, tell me and I'll factor "
                    "it in. You can also ask me anything about it, like what tends to help.")
        return {**out, "answer": text, "state": state, "assessment": _assessment(state, matches, [], True, assessor)}

    # Greetings, or "I want to check my symptoms" with nothing yet.
    if not found and not state["present"] and info["intent"] in ("other", "assess") and not state["pending"]:
        state["mode"] = "assess"
        if llm_configured():
            text = write(history, message, (
                "The person wants to go through their symptoms but hasn't described any yet. Invite them warmly to tell "
                "you what's been going on in their own words, with two or three examples of helpful things to mention "
                "(timing in the cycle, pain, flow, mood, skin). Under 60 words. No list."))
        else:
            text = _fallback_reply("open", state, [], [], rng, True)
            if GREET(message):
                text = "Hi, I'm Luna. " + text.replace("Of course. ", "", 1)
        return {**out, "answer": text, "state": state}

    first = not state["present"]
    was_concluded = state["concluded"]
    before = assessor.score(state)
    prev_top = before[0]["id"] if before else None
    prev_top3 = {m["id"] for m in before[:3]}
    assessor.apply(state, info["present"], info["absent"])
    state["asked"] = list(dict.fromkeys(state["asked"] + state["pending"]))
    state["pending"] = []
    state["mode"] = "assess"

    matches = assessor.score(state)
    questions = assessor.next_questions(state, matches)
    ready = assessor.is_ready(state, matches, questions, info["intent"]) or (was_concluded and bool(matches))
    urgent = assessor.urgent(state)
    if not matches:          # nothing recognisable yet: ask broad starters
        ready = False
    if ready:
        state["concluded"] = True
    else:
        state["pending"] = questions
        state["rounds"] += 1
    qtexts = [assessor.symptoms[q]["question"] for q in state["pending"]]

    chunks = []
    if ready:
        labels = ", ".join(assessor.symptoms[s]["label"] for s in state["present"])
        seen = set()
        for m in matches[:3]:
            for c in kb.retrieve(f"{m['name']} symptoms: {labels}", k=2, urls=m["sources"], min_score=0.0):
                if c["id"] not in seen:
                    seen.add(c["id"]); chunks.append(c)

    if llm_configured():
        have = ", ".join(assessor.symptoms[s]["label"] for s in state["present"]) or "nothing yet"
        dont = ", ".join(assessor.symptoms[s]["label"] for s in state["absent"]) or "nothing"
        safety = (f"\nA safety message is already shown in red above your reply: {' '.join(urgent)} Open with one short, "
                  "caring sentence that backs it up, then keep the rest brief.") if urgent else ""
        if ready and was_concluded:
            same = matches[0]["id"] == prev_top
            notes = (f"Symptoms they have: {have}\nSymptoms they don't have: {dont}\nYou already shared a result earlier. "
                     f"They've just told you something new. With it, the best fit is {'still ' if same else 'now '}"
                     f"{matches[0]['name']} ({matches[0]['percent']}%)" + ("" if same else f", ahead of what you said before") +
                     f".\nTop patterns now: " + "; ".join(f"{m['name']} {m['percent']}%" for m in matches[:3]) +
                     ". If a pattern has newly appeared near the top, mention it as another possibility (people can have "
                     f"more than one thing going on).\nExcerpts:\n{_context_block(chunks)}\n\nAcknowledge the new detail naturally and say in a few "
                     "sentences how it changes, or doesn't change, the picture, citing an excerpt if useful. Don't repeat "
                     f"your whole earlier answer. Under 110 words.{safety}")
        elif ready:
            ranked = "\n".join(f"- {m['name']}: {m['percent']}%. Fits: {', '.join(m['matched']) or 'none'}."
                               + (f" Doesn't fit: {', '.join(m['against'])}." if m["against"] else "")
                               for m in matches[:3])
            notes = (f"Symptoms they have: {have}\nSymptoms they don't have: {dont}\nRanked patterns (do not change the order):\n"
                     f"{ranked}\nExcerpts:\n{_context_block(chunks)}\n\nYou now know enough to share what it looks like. "
                     "In warm, flowing paragraphs: say which pattern fits best and roughly how well (you may give the "
                     "percentage once), explaining why with specific things they told you, and cite the excerpts. "
                     "Briefly mention one or two other possibilities and what makes them less likely. Suggest what a GP or "
                     "gynaecologist might check. Make clear, kindly and in your own words, that this is a comparison with "
                     "common patterns and not a diagnosis. End by inviting questions, for example about what helps. "
                     f"Under 230 words.{safety}")
        else:
            lean = (f"It currently leans towards {matches[0]['name']} but it's too early to say. You may hint at that "
                    "tentatively in a few words, or not at all.") if matches and matches[0]["score"] >= 0.45 and not first else \
                   "Don't guess at a condition yet."
            notes = (f"Symptoms they have: {have}\nSymptoms they don't have: {dont}\n{lean}\n"
                     f"Ask about these next, in your own words: {' | '.join(qtexts)}\n\n"
                     "Reply to their latest message like a caring person would. React briefly and specifically to what "
                     "they just said (one sentence, no list of their symptoms). Then ask those questions naturally in "
                     "flowing text, joined into one or two sentences, not a numbered list. Under 75 words. No citations."
                     f"{safety}")
        text = write(history, message, notes)
    else:
        if ready and was_concluded:
            top = matches[0]
            newcomer = next((m for m in matches[1:3] if m["id"] not in prev_top3 and m["percent"] >= 35), None)
            if top["id"] == prev_top and newcomer:
                text = (f"Thanks, that's helpful. It still fits {top['short']} best, but those new details also point "
                        f"towards {newcomer['short']}. {newcomer['summary']} It's possible to have more than one thing going "
                        "on, so it's worth mentioning both to your doctor.")
            elif top["id"] == prev_top:
                text = (f"Thanks, I've added that in. It still fits {top['short']} best, and that detail doesn't change the "
                        "overall picture much. It's worth mentioning to your doctor too.")
            else:
                text = (f"Thanks, that's an important detail and it shifts things a little. It now fits {top['name']} best. "
                        f"{top['summary']} Still not a diagnosis, but worth raising with a GP or gynaecologist.")
        else:
            text = _fallback_reply("result" if ready else "ask", state, matches, qtexts, rng, first)

    return {**out, "answer": text, "sources": _sources(chunks) if ready else [], "urgent": urgent, "state": state,
            "assessment": _assessment(state, matches, qtexts, ready, assessor), "extractor": info["extractor"]}


def _assessment(state, matches, qtexts, ready, assessor):
    return {"matches": matches[:5], "questions": qtexts, "ready": ready, "rounds": state["rounds"],
            "symptoms": assessor.picture(state), "completeness": assessor.completeness(state, matches),
            "concluded": state.get("concluded", False)}


__all__ = ["KnowledgeBase", "Assessor", "empty_state", "symptom_turn", "qa_turn", "llm_configured"]
