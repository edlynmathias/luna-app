"""Luna: a menstrual-health guide with two sections, a symptom check and general questions.
Run locally:  streamlit run streamlit_app.py
"""
import html, os, sys, time
from pathlib import Path

# All paths are built from this file's folder, so the app works from the repo root or a subfolder.
# (Don't os.chdir here: Streamlit Cloud re-runs the script by a path relative to the repo root.)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st

try:  # only needed for local runs with a .env file
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=True)
except ImportError:
    pass
try:  # Streamlit Cloud: App settings > Secrets
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

from rag import KnowledgeBase, Assessor, symptom_turn, qa_turn, empty_state, llm_configured

st.set_page_config(page_title="Luna", page_icon=":material/bedtime:", layout="centered",
                   initial_sidebar_state="auto")

SYMPTOMS, QA = "Symptom check", "Ask about periods"
LUNA_AVATAR, USER_AVATAR = ":material/bedtime:", ":material/favorite:"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Nunito:wght@400;500;600;700&display=swap');
:root { --blush:#FFF3F7; --petal:#FFE1EB; --rose:#E2598B; --berry:#A8325F; --plum:#47202F; --mauve:#8E6070; --line:#F5C9D8; }
.stApp { background: radial-gradient(1200px 600px at 50% -10%, #FFDDE9 0%, #FFF3F7 55%, #FFF8FA 100%); color: var(--plum); font-family: "Nunito", "Segoe UI", sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input, .stApp button { font-family: "Nunito", "Segoe UI", sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 7rem; max-width: 760px; }
header[data-testid="stHeader"] { background: transparent; }

/* header */
.luna-head { display:flex; align-items:center; gap:14px; margin-bottom: 4px; }
.luna-head h1 { font-family:"Fraunces", Georgia, serif; font-weight:500; font-size: 2.6rem; color: var(--plum); margin:0; padding:0; letter-spacing:-.01em; }
.luna-tag { color: var(--mauve); margin: 0 0 1.2rem 0; font-size: 1.02rem; }

/* section switch */
button[data-testid="stBaseButton-segmented_control"], button[data-testid="stBaseButton-segmented_controlActive"] {
  border: 1px solid var(--line) !important; background: #fff !important; color: var(--plum) !important; padding: .4rem 1.2rem !important; font-weight: 600; }
button[data-testid="stBaseButton-segmented_control"]:first-child, button[data-testid="stBaseButton-segmented_controlActive"]:first-child { border-radius: 999px 0 0 999px !important; }
button[data-testid="stBaseButton-segmented_control"]:last-child, button[data-testid="stBaseButton-segmented_controlActive"]:last-child { border-radius: 0 999px 999px 0 !important; }
button[data-testid="stBaseButton-segmented_controlActive"] { background: linear-gradient(135deg, #F07AA6, #D94A7E) !important; color: #fff !important; border-color: #D94A7E !important; }
button[data-testid="stBaseButton-segmented_controlActive"] p { color: #fff !important; }

/* hero for an empty chat */
.hero { background: rgba(255,255,255,.72); border: 1px solid var(--line); border-radius: 22px; padding: 22px 24px 8px; margin: .4rem 0 .8rem; }
.hero h2 { font-family:"Fraunces", Georgia, serif; font-weight:500; font-size: 1.65rem; line-height:1.2; color: var(--plum); margin:0 0 .4rem; padding:0; }
.stApp .hero p { color: var(--mauve); margin: 0 0 .8rem; }

/* starter chips */
div[data-testid="stHorizontalBlock"] .stButton > button, .chip .stButton > button { border-radius: 999px; border: 1px solid var(--line); background: #fff; color: var(--plum); font-weight: 600; }
.stButton > button:hover { border-color: var(--rose); color: var(--berry); }

/* chat */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: .8rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p, [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { font-size: 1.04rem; line-height: 1.65; }
[data-testid="stChatMessageAvatarCustom"] { background: linear-gradient(135deg, #F7A1C1, #D94A7E); border: 0; border-radius: 50%; }
[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] { color: #fff !important; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] { background: var(--petal); border: 1px solid var(--line); }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] { color: var(--rose) !important; }
[aria-label="Chat message from user"] { text-align: right; }
.user-bubble { background: var(--petal); border: 1px solid var(--line); border-radius: 18px 18px 6px 18px; padding: .6rem 1rem; display:inline-block; color: var(--plum); text-align: left; max-width: 85%; }
[data-testid="stChatMessage"] a { color: var(--berry); }
.urgent { background:#FDE2E6; border:1px solid #F0A1AF; border-left:4px solid #D9485F; color:#6E1A2B; border-radius: 12px; padding:.75rem 1rem; margin:.2rem 0 .6rem; font-weight:600; }
.srcs { font-size:.85rem; color: var(--mauve); margin-top:.2rem; }
.srcs a { color: var(--mauve); text-decoration: none; display:block; margin: .12rem 0; }
.srcs a:hover { color: var(--berry); text-decoration: underline; }
.srcs b { color: var(--berry); margin-right: .35rem; }
[data-testid="stChatInput"] { border-radius: 18px; }
[data-testid="stChatInput"] textarea { font-size: 1rem; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div { background: transparent; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFE8F0 0%, #FFF4F8 100%); border-right: 1px solid var(--line); }
.moon-wrap { display:flex; align-items:center; gap:14px; margin: .4rem 0 1rem; }
.moon-wrap svg { flex: none; width: 64px; height: 64px; }
.moon-wrap b { display:block; color: var(--plum); font-size: 1rem; }
.moon-wrap span { color: var(--mauve); font-size: .86rem; line-height: 1.35; display:block; }
.side-h { font-family:"Fraunces", Georgia, serif; font-weight:500; font-size: 1.05rem; color: var(--plum); margin: 1.1rem 0 .45rem; }
.tag { display:inline-block; font-size:.84rem; padding:3px 11px; border-radius: 999px; background:#fff; border:1px solid var(--line); color: var(--plum); margin:0 5px 7px 0; }
.tag.no { background: transparent; color: var(--mauve); text-decoration: line-through; }
.m-row { margin-bottom: .55rem; }
.m-head { display:flex; justify-content:space-between; gap:10px; font-size:.9rem; color: var(--plum); }
.m-head span:last-child { color: var(--mauve); font-variant-numeric: tabular-nums; }
.m-row.lead .m-head span:first-child { font-weight: 700; }
.bar { height:7px; background:#FBD9E4; border-radius: 99px; margin-top:5px; overflow:hidden; }
.bar i { display:block; height:100%; background: linear-gradient(90deg, #F48FB1, var(--rose)); border-radius: 99px; }
.stApp p.fine, .fine { font-size:.82rem !important; color: var(--mauve); line-height:1.45; }
.empty { font-size:.88rem; color: var(--mauve); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Opening Luna's health library...")
def load():
    kb = KnowledgeBase()
    kb.ingest(HERE / "knowledge" / "sources.json")
    return kb, Assessor()


kb, assessor = load()
ss = st.session_state
ss.setdefault("section", SYMPTOMS)
ss.setdefault("last_section", SYMPTOMS)
ss.setdefault(SYMPTOMS, {"entries": [], "history": [], "state": empty_state(), "assessment": None})
ss.setdefault(QA, {"entries": [], "history": [], "state": {}})


# ---------------------------------------------------------------- pieces
def moon_svg(f, size=64):
    """A moon whose lit part grows with f (0 = new moon, 1 = full)."""
    r, c = size / 2 - 3, size / 2
    rx, sweep = abs(1 - 2 * f) * r, 0 if f < .5 else 1
    if f <= .001:
        lit = ""
    elif f >= .999:
        lit = f'<circle cx="{c}" cy="{c}" r="{r}" fill="url(#lit)"/>'
    else:
        lit = f'<path d="M{c} {c - r} A{r} {r} 0 0 1 {c} {c + r} A{rx} {r} 0 0 {sweep} {c} {c - r}Z" fill="url(#lit)"/>'
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-hidden="true">'
            '<defs><radialGradient id="lit" cx="40%" cy="35%" r="70%"><stop offset="0" stop-color="#FFFFFF"/>'
            '<stop offset="1" stop-color="#FFE4EE"/></radialGradient></defs>'
            f'<circle cx="{c}" cy="{c}" r="{r + 2}" fill="#FFD3E2" opacity=".45"/>'
            f'<circle cx="{c}" cy="{c}" r="{r}" fill="#F6C1D3"/>{lit}'
            f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#EFA9C1" stroke-width="1"/></svg>')


CRESCENT = ('<svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><defs><linearGradient id="cg" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#F7A1C1"/><stop offset="1" stop-color="#C83F72"/></linearGradient></defs>'
            '<path d="M25 4a16 16 0 1 0 11 26A13.5 13.5 0 0 1 25 4z" fill="url(#cg)"/></svg>')


def bars(matches, lead_first=True):
    return "".join(f'<div class="m-row{" lead" if lead_first and i == 0 else ""}"><div class="m-head"><span>{html.escape(m["name"])}</span>'
                   f'<span>{m["percent"]}%</span></div><div class="bar"><i style="width:{m["percent"]}%"></i></div></div>'
                   for i, m in enumerate(matches))


def sources_html(sources):
    return '<div class="srcs">' + "".join(
        f'<a href="{html.escape(s["url"])}" target="_blank" rel="noopener"><b>{s["n"]}</b>{html.escape(s["title"])}</a>'
        for s in sources) + "</div>"


def show_entry(e):
    if e["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(f'<div class="user-bubble">{html.escape(e["content"])}</div>', unsafe_allow_html=True)
        return
    with st.chat_message("assistant", avatar=LUNA_AVATAR):
        render_luna(e, stream=False)


def render_luna(e, stream):
    for msg in e.get("urgent") or []:
        st.markdown(f'<div class="urgent">{html.escape(msg)}</div>', unsafe_allow_html=True)
    if stream:
        def words(text):
            for i, w in enumerate(text.split(" ")):
                yield w + " "
                time.sleep(0.018 if i < 400 else 0)
        st.write_stream(words(e["content"]))
    else:
        st.markdown(e["content"])
    if e.get("sources"):
        with st.expander("Where this comes from"):
            st.markdown(sources_html(e["sources"]), unsafe_allow_html=True)
    a = e.get("assessment")
    if e.get("result") and a and a["matches"]:
        with st.expander("How your symptoms compare"):
            st.markdown(bars(a["matches"][:4]), unsafe_allow_html=True)
            st.markdown('<p class="fine">How closely your answers fit each pattern. Not the chance that you have it.</p>',
                        unsafe_allow_html=True)


THINKING = {SYMPTOMS: ["Luna is thinking about what you said...", "Luna is going through the patterns...",
                       "Luna is checking her sources..."],
            QA: ["Luna is looking through NHS and MedlinePlus pages...", "Luna is checking her sources..."]}


def run_turn(section, text):
    box = ss[section]
    user = {"role": "user", "content": text}
    box["entries"].append(user)
    show_entry(user)
    with st.chat_message("assistant", avatar=LUNA_AVATAR):
        try:
            with st.spinner(THINKING[section][len(box["entries"]) % len(THINKING[section])]):
                if section == SYMPTOMS:
                    r = symptom_turn(text, box["history"], box["state"], kb, assessor)
                else:
                    r = qa_turn(text, box["history"], box["state"], kb)
            a = r.get("assessment")
            entry = {"role": "assistant", "content": r["answer"], "sources": r.get("sources", []),
                     "urgent": r.get("urgent", []), "assessment": a,
                     "result": bool(a and a.get("ready") and r.get("sources"))}
            box["state"] = r["state"]
            if a:
                box["assessment"] = a
            box["history"] = (box["history"] + [{"role": "user", "content": text},
                                                {"role": "assistant", "content": r["answer"]}])[-28:]
        except Exception as ex:
            msg = ("I'm getting a lot of requests right now, so give me a minute and try again."
                   if "429" in str(ex) else f"Something went wrong on my side ({ex}). Could you try again?")
            entry = {"role": "assistant", "content": msg}
        render_luna(entry, stream=True)
        box["entries"].append(entry)


# ---------------------------------------------------------------- page
st.markdown(f'<div class="luna-head">{CRESCENT}<h1>Luna</h1></div>'
            '<p class="luna-tag">A gentle guide to your period, grounded in NHS, MedlinePlus and ACOG advice.</p>',
            unsafe_allow_html=True)

def keep_a_section():
    # Clicking the selected section again would leave nothing selected; keep the last one instead.
    if ss.section is None:
        ss.section = ss.last_section
    ss.last_section = ss.section


section = st.segmented_control("Section", [SYMPTOMS, QA], key="section", label_visibility="collapsed",
                               on_change=keep_a_section) or ss.last_section
box = ss[section]

STARTERS = {
    SYMPTOMS: ("What's been going on?",
               "Tell me what you're feeling in your own words. I'll ask a few gentle questions, remember everything you "
               "say, and then tell you which condition it sounds closest to.",
               ["My periods are really irregular", "I get really bad cramps", "My periods are very heavy",
                "I feel low and irritable before my period"]),
    QA: ("Ask me anything about periods",
         "From easing cramps to what's normal for your cycle. I'll answer from trusted health sources and show you "
         "where it came from.",
         ["How can I ease period cramps?", "Is it normal for my cycle to change?", "What helps with bloating?",
          "When should I see a doctor about my period?"]),
}

for e in box["entries"]:
    show_entry(e)

placeholder = ("Describe what you're feeling, or answer Luna's question" if section == SYMPTOMS
               else "Ask about periods, cramps, products, your cycle...")
typed = st.chat_input(placeholder, key=f"input-{section}")   # pinned to the bottom wherever it's called
pending = ss.pop("pending_msg", None)
if not box["entries"] and not (typed or pending):
    title, sub, chips = STARTERS[section]
    st.markdown(f'<div class="hero"><h2>{title}</h2><p>{sub}</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, chip in enumerate(chips):
        if cols[i % 2].button(chip, key=f"chip-{section}-{i}", use_container_width=True):
            ss.pending_msg = chip
            st.rerun()

if typed or pending:
    run_turn(section, typed or pending)


# ---------------------------------------------------------------- sidebar (drawn last so it shows this turn's state)
with st.sidebar:
    if section == SYMPTOMS:
        a = ss[SYMPTOMS]["assessment"]
        present = a["symptoms"]["present"] if a else []
        f = a["completeness"] if a else 0
        if not present:
            title, sub = "A new moon", "Tell Luna what you're feeling and the moon will start to fill."
        elif a.get("concluded"):
            f = max(f, 1.0)
            title, sub = "Full moon", "Luna has shared what your symptoms sound closest to."
        else:
            title, sub = f"{round(f * 100)}% of the picture", "Each answer helps Luna understand you better."
        st.markdown(f'<div class="moon-wrap">{moon_svg(f)}<div><b>{title}</b><span>{sub}</span></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="side-h">What you\'ve told Luna</div>' + ("".join(
            f'<span class="tag">{html.escape(s["label"])}</span>' for s in present) or
            '<span class="empty">Nothing yet.</span>'), unsafe_allow_html=True)
        absent = a["symptoms"]["absent"] if a else []
        if absent:
            st.markdown('<div class="side-h">Not something you have</div>' + "".join(
                f'<span class="tag no">{html.escape(s["label"])}</span>' for s in absent), unsafe_allow_html=True)
        if a and a["matches"]:
            st.markdown('<div class="side-h">Closest patterns so far</div>' + bars(a["matches"][:4]), unsafe_allow_html=True)
        st.write("")
        if st.button("Start a new symptom check", use_container_width=True):
            ss[SYMPTOMS] = {"entries": [], "history": [], "state": empty_state(), "assessment": None}
            st.rerun()
    else:
        st.markdown(f'<div class="moon-wrap">{moon_svg(1.0)}<div><b>Ask Luna</b><span>Every answer comes from trusted '
                    'health sources, with links so you can read more.</span></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-h">Good to know</div><p class="fine">Get urgent help for sudden severe pain, '
                    'fainting, bleeding through a pad every hour for several hours, or a fever with a rash while using a '
                    'tampon or cup.</p>', unsafe_allow_html=True)
        st.write("")
        if st.button("Clear this chat", use_container_width=True):
            ss[QA] = {"entries": [], "history": [], "state": {}}
            st.rerun()
    st.markdown('<p class="fine" style="margin-top:1.2rem">Luna is for information only and is not a diagnosis. '
                f'{"" if llm_configured() else "AI writing is off: add LLM_API_KEY in Secrets for full replies. "}'
                f'Search: {"semantic" if kb.mode.startswith("semantic") else "keyword"}.</p>', unsafe_allow_html=True)
