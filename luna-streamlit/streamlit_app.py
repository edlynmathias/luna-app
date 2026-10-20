"""Streamlit version of Luna (for Streamlit Community Cloud). Run: streamlit run streamlit_app.py"""
import html, os, sys
from pathlib import Path

# Work from this file's folder, so knowledge/ and .chroma are found even if the app sits in a subfolder of the repo.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

# Streamlit Cloud's system sqlite can be too old for Chroma; use the bundled one if available.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
try:  # python-dotenv is only needed for local runs with a .env file
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=True)
except ImportError:
    pass
try:  # On Streamlit Cloud, keys come from App settings -> Secrets (same names as .env).
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

from rag import KnowledgeBase, Assessor, chat_turn, empty_state, llm_configured

st.set_page_config(page_title="Luna, menstrual health guide", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root { --paper:#EEECF1; --surface:#F8F7FA; --ink:#221A2B; --muted:#6B6274; --line:#D9D4DE; --accent:#8A2846; --soft:#EBDDE3; }
html, body, .stApp, [class*="css"] { font-family: "Hanken Grotesk", "Segoe UI", system-ui, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 760px; padding-top: 2.2rem; }
section[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
h1.luna { font-family: "Newsreader", Georgia, serif; font-weight: 400; font-size: 2.6rem; line-height: 1.1;
          letter-spacing: -.015em; margin: 0 0 .4rem; color: var(--ink); }
p.lede { color: var(--muted); margin: 0 0 1.4rem; max-width: 56ch; }
[data-testid="stChatMessage"] { background: transparent; padding: .4rem 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li {
  font-family: "Newsreader", Georgia, serif; font-size: 1.12rem; line-height: 1.6; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px; padding: .5rem 1rem; }
.tag { display:inline-block; font-size:.82rem; padding:2px 9px; border-radius:6px; background:var(--soft); margin:0 4px 6px 0; }
.tag.no { background:transparent; border:1px solid var(--line); color:var(--muted); text-decoration:line-through; }
.m-head { display:flex; justify-content:space-between; font-size:.9rem; margin-top:.6rem; }
.m-head span:last-child { color:var(--muted); font-variant-numeric: tabular-nums; }
.bar { height:5px; background:var(--line); border-radius:3px; margin-top:5px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--accent); }
.urgent { background:#F6E3E6; border-left:3px solid #C9677D; padding:.7rem 1rem; border-radius:4px; margin-bottom:.6rem; }
.stButton > button, .stFormSubmitButton > button { border-radius: 8px; font-weight: 600; }
.sub { font-weight:600; font-size:.92rem; margin:1.1rem 0 .4rem; }
.fine { font-size:.8rem; color:var(--muted); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Luna's health library...")
def load():
    kb = KnowledgeBase()
    kb.ingest(HERE / "knowledge" / "sources.json")
    return kb, Assessor()


kb, assessor = load()
ss = st.session_state
ss.setdefault("entries", [])       # what's shown on screen
ss.setdefault("history", [])       # what's sent to the LLM as conversation memory
ss.setdefault("state", empty_state())
ss.setdefault("assessment", None)


def moon_svg(f, size=56):
    r, c = size / 2 - 2, size / 2
    rx, sweep = abs(1 - 2 * f) * r, 0 if f < .5 else 1
    lit = "" if f <= .001 else (f'<circle cx="{c}" cy="{c}" r="{r}" fill="#FBFAF7"/>' if f >= .999 else
          f'<path d="M{c} {c - r} A{r} {r} 0 0 1 {c} {c + r} A{rx} {r} 0 0 {sweep} {c} {c - r}Z" fill="#FBFAF7"/>')
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"><circle cx="{c}" cy="{c}" r="{r}" fill="#D6D1DC"/>'
            f'{lit}<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#C9C2CF"/></svg>')


def bars(matches):
    return "".join(f'<div class="m-head"><span>{html.escape(m["name"])}</span><span>{m["percent"]}%</span></div>'
                   f'<div class="bar"><i style="width:{m["percent"]}%"></i></div>' for m in matches)


# ---------------- sidebar: the symptom picture ----------------
with st.sidebar:
    a = ss.assessment
    f = a["completeness"] if a else 0
    present = a["symptoms"]["present"] if a else []
    title = ("No symptoms yet" if not present else "Picture complete enough" if a["ready"]
             else f"Picture {round(f * 100)}% filled in")
    st.markdown(f'<div style="display:flex;gap:14px;align-items:center">{moon_svg(f)}'
                f'<div><b>{title}</b><div class="fine">How much of the leading match Luna has checked.</div></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub">Symptoms you have</div>' + ("".join(
        f'<span class="tag">{html.escape(s["label"])}</span>' for s in present) or '<span class="fine">None noted yet.</span>'),
        unsafe_allow_html=True)
    absent = a["symptoms"]["absent"] if a else []
    st.markdown('<div class="sub">Ruled out</div>' + ("".join(
        f'<span class="tag no">{html.escape(s["label"])}</span>' for s in absent) or '<span class="fine">Nothing yet.</span>'),
        unsafe_allow_html=True)
    st.markdown('<div class="sub">Closest matches</div>' + (bars(a["matches"][:4]) if a and a["matches"] else
                '<span class="fine">Matches appear once Luna knows a symptom.</span>'), unsafe_allow_html=True)
    st.write("")
    if st.button("Start a new conversation", use_container_width=True):
        for k in ("entries", "history", "state", "assessment", "pending"):
            ss.pop(k, None)
        st.rerun()
    st.markdown(f'<p class="fine">Search: {"semantic" if kb.mode.startswith("semantic") else "keyword"}. '
                f'AI answers: {"on" if llm_configured() else "off"}. {len(kb.docs)} source sections. '
                'Luna compares symptoms with common patterns from NHS, MedlinePlus, ACOG and similar sources. '
                'It is not a diagnosis.</p>', unsafe_allow_html=True)


def run_turn(text, answers=None, general=False, display=None):
    ss.entries.append({"role": "user", "content": display or text})
    try:
        with st.spinner("Checking trusted sources..."):
            r = chat_turn(text, ss.history, ss.state, kb, assessor, answers=answers, force_general=general)
        ss.history = (ss.history + [{"role": "user", "content": text},
                                    {"role": "assistant", "content": r["history_text"]}])[-24:]
        ss.state = r["state"]
        if r.get("assessment"):
            ss.assessment = r["assessment"]
        ss.entries.append({"role": "assistant", "content": r["answer"], "sources": r["sources"],
                           "assessment": r.get("assessment"), "offer": r.get("offer_general"), "question": text})
    except Exception as e:
        msg = ("The AI provider is rate-limiting requests. Wait a minute and try again." if "429" in str(e)
               else f"The AI provider returned an error: {e}")
        ss.entries.append({"role": "assistant", "content": msg, "sources": []})


# ---------------- main column ----------------
if not ss.entries:
    st.markdown('<h1 class="luna">Tell Luna what your body is doing.</h1>'
                '<p class="lede">Describe your symptoms in your own words. Luna asks follow-up questions, remembers what '
                "you've said, and shows which condition your symptoms match most closely, with the sources behind it. "
                'You can also just ask a question about periods.</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, q in zip(cols, ["Check my symptoms", "Is my period too heavy?", "What is PCOS?", "Are bad cramps normal?"]):
        if col.button(q, use_container_width=True, type="primary" if q.startswith("Check") else "secondary"):
            ss.pending = "I'd like to check my symptoms" if q.startswith("Check") else q
            st.rerun()

for i, e in enumerate(ss.entries):
    last = i == len(ss.entries) - 1
    with st.chat_message(e["role"]):
        a = e.get("assessment")
        text = e["content"]
        if a and a.get("urgent") and text.startswith("Please act on this first:"):
            head, _, text = text.partition("\n\n")
            st.markdown(f'<div class="urgent">{html.escape(head)}</div>', unsafe_allow_html=True)
        st.markdown(text)
        if e.get("sources"):
            st.caption("Sources: " + "  \n".join(f"{s['n']}. [{s['title']}]({s['url']})" for s in e["sources"]))
        if a and a.get("ready") and a.get("matches"):
            top = a["matches"][0]
            st.markdown(f"**{top['name']}** &nbsp; {top['percent']}% match  \n{top['summary']}")
            st.markdown(bars(a["matches"][1:4]), unsafe_allow_html=True)
        if last and a and a.get("questions"):
            with st.form(f"q{i}"):
                picks = {}
                for q in a["questions"]:
                    picks[q["id"]] = st.radio(q["text"], ["Yes", "No", "Not sure"], index=None, horizontal=True, key=f"{i}-{q['id']}")
                if st.form_submit_button("Send answers", type="primary"):
                    answered = {k: v for k, v in picks.items() if v}
                    if answered:
                        label = {q["id"]: q["label"].lower() for q in a["questions"]}
                        groups = {}
                        for k, v in answered.items():
                            groups.setdefault(v, []).append(label[k])
                        summary = ". ".join(f"{k}: {', '.join(v)}" for k, v in groups.items()) + "."
                        ss.pending_answers = ({k: v == "Yes" for k, v in answered.items() if v != "Not sure"}, summary)
                        st.rerun()
            st.caption("Or reply in your own words below.")
        if last and e.get("offer"):
            if st.button("Answer without sources"):
                ss.pending_general = e["question"]
                st.rerun()

st.caption("Educational information only. For sudden severe pain, fainting or very heavy bleeding, contact emergency services.")

q = st.chat_input("Describe your symptoms or ask a question")
if q:
    run_turn(q); st.rerun()
elif ss.get("pending"):
    run_turn(ss.pop("pending")); st.rerun()
elif ss.get("pending_answers"):
    answers, summary = ss.pop("pending_answers")
    run_turn(summary, answers=answers); st.rerun()
elif ss.get("pending_general"):
    run_turn(ss.pop("pending_general"), general=True, display="Answer without sources"); st.rerun()
