"""Symptom assessment engine for Luna.

Turns what the user has said into present / absent symptoms, scores every condition in knowledge/conditions.json,
and picks the follow-up questions that would best tell the leading candidates apart (expected information gain).
The ranking is deterministic and explainable: the LLM only puts it into words, it never decides it.
"""
import json, math, re
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "knowledge" / "conditions.json"

NEGATION = re.compile(r"\b(no|not|never|don'?t|doesn'?t|didn'?t|haven'?t|hasn'?t|isn'?t|aren'?t|without|nor|none)\b|n't\b")
YES = re.compile(r"^\s*(yes|yeah|yep|yup|ya|y|sure|i do|it does|they do|definitely|correct|right|kind of|sometimes|a bit|a little)\b", re.I)
NO = re.compile(r"^\s*(no|nope|nah|n|i don'?t|it doesn'?t|not really|never|neither)\b", re.I)
ALL_YES = re.compile(r"\b(yes to (all|both|everything)|all of (them|those)|both of (them|those)|both)\b", re.I)
ALL_NO = re.compile(r"\b(no to (all|both|everything)|none of (them|those|these)|neither)\b", re.I)
ORDINAL = re.compile(r"\b(yes|yeah|yep|no|nope|not)\b[^.,;]*?\b(first|1st|second|2nd|third|3rd)\b"
                     r"|\b(first|1st|second|2nd|third|3rd)\b[^.,;]*?\b(yes|yeah|yep|no|nope|not)\b", re.I)
ORD_INDEX = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
RESULTS = re.compile(r"\b(result|results|what do i have|what could (it|this) be|what is it|which (condition|syndrome|disorder)|"
                     r"diagnos\w*|that'?s (all|it|everything)|nothing else|just tell me|what matches|conclusion|what'?s wrong)\b", re.I)
QUESTION_START = re.compile(r"^\s*(what|how|why|when|where|which|who|can|could|should|is|are|do|does|will|tell me about)\b", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|hiya|heya|good (morning|afternoon|evening)|thanks|thank you|ok(ay)?|cool)\b[\s!.]*$", re.I)
START_ASSESS = re.compile(r"\b((check|assess|go through) my symptoms|symptom check|what'?s wrong with me|"
                          r"i (have|'ve got|get) (some |a few |these )?symptoms|which condition (do i|might i))\b", re.I)
CRISIS = re.compile(r"\b(suicid\w*|kill (myself|me)|end my life|self[- ]?harm|hurt myself|don'?t want to (live|be alive))\b", re.I)

QUESTIONS_PER_TURN = 2
MAX_ROUNDS = 5            # follow-up rounds before Luna always shares what it thinks
READY_SCORE = 0.50        # top match needed to stop early ...
READY_MARGIN = 0.12       # ... with this lead over the runner-up


def empty_state():
    return {"present": [], "absent": [], "asked": [], "pending": [], "rounds": 0, "mode": "qa", "concluded": False}


class Assessor:
    def __init__(self, path=DEFAULT_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.symptoms = {s["id"]: s for s in data["symptoms"]}
        self.conditions = data["conditions"]
        self.by_id = {c["id"]: c for c in self.conditions}
        self.red_flags = data["red_flags"]
        self._patterns = {sid: [re.compile(p, re.I) for p in s["synonyms"]] for sid, s in self.symptoms.items()}

    # ------------------------------------------------------------------ state
    def sanitize(self, state):
        base = empty_state()
        if not isinstance(state, dict):
            return base
        for key in ("present", "absent", "asked", "pending"):
            vals = state.get(key) or []
            if isinstance(vals, list):
                base[key] = list(dict.fromkeys(v for v in vals if isinstance(v, str) and v in self.symptoms))
        base["absent"] = [s for s in base["absent"] if s not in base["present"]]
        try:
            base["rounds"] = max(0, min(int(state.get("rounds", 0)), 20))
        except (TypeError, ValueError):
            pass
        base["mode"] = "assess" if state.get("mode") == "assess" else "qa"
        base["concluded"] = bool(state.get("concluded"))
        return base

    def apply(self, state, present=(), absent=()):
        """Merge new findings. The newest answer for a symptom wins."""
        P, A = list(state["present"]), list(state["absent"])
        for s in present:
            if s in self.symptoms:
                if s in A: A.remove(s)
                if s not in P: P.append(s)
        for s in absent:
            if s in self.symptoms:
                if s in P: P.remove(s)
                if s not in A: A.append(s)
        state["present"], state["absent"] = P, A
        return state

    # ------------------------------------------------------------ extraction
    def extract_keywords(self, text, pending=()):
        """Rule-based extraction used when no LLM is configured, or the LLM call fails."""
        present, absent = set(), set()
        pending = [p for p in pending if p in self.symptoms]
        short = len(text.split()) <= 8
        if pending:
            ords = list(ORDINAL.finditer(text))
            if ords:
                for m in ords:
                    word = (m.group(1) or m.group(4)).lower()
                    idx = ORD_INDEX[(m.group(2) or m.group(3)).lower()]
                    if idx < len(pending):
                        (absent if word in ("no", "nope", "not") else present).add(pending[idx])
            elif ALL_NO.search(text) or (short and NO.search(text)):
                absent.update(pending)
            elif ALL_YES.search(text) or (short and YES.search(text)):
                present.update(pending)      # questions are asked together, so a bare yes covers them
        for clause in re.split(r"[.;!?\n]|,\s*(?:but|and)\s|\bbut\b", text.lower()):
            for sid, pats in self._patterns.items():
                for p in pats:
                    m = p.search(clause)
                    if not m:
                        continue
                    before = " ".join(clause[:m.start()].split()[-4:])
                    (absent if NEGATION.search(before) else present).add(sid)
                    break
        absent -= present
        return sorted(present), sorted(absent)

    def detect_intent(self, text, state, found):
        if CRISIS.search(text):
            return "crisis"
        if GREETING.search(text):
            return "other"
        if RESULTS.search(text) and (state["present"] or found):
            return "results"
        if found:
            return "assess"
        if QUESTION_START.search(text) or text.strip().endswith("?"):
            return "question"
        if START_ASSESS.search(text) or state["pending"]:
            return "assess"
        return "other"

    # ---------------------------------------------------------------- scoring
    def score(self, state):
        """match = 0.55 x profile fit + 0.45 x share of the user's symptoms the condition explains."""
        P, A = set(state["present"]), set(state["absent"])
        real = [s for s in P if self.symptoms[s]["kind"] == "symptom"]
        out = []
        for c in self.conditions:
            prof, against = c["profile"], c.get("against", {})
            support = sum(w for s, w in prof.items() if s in P)
            if support == 0:
                continue
            contra = sum(0.5 * w for s, w in prof.items() if s in A) + sum(w for s, w in against.items() if s in P)
            fit = max(0.0, support - contra) / sum(prof.values())
            explained = (sum(1 for s in real if s in prof) / len(real)) if real else 0.5
            match = 0.55 * fit + 0.45 * explained
            key = c.get("key", [])
            if key and not any(k in P for k in key):
                match *= 0.3 if all(k in A for k in key) else 0.6
            out.append({
                "id": c["id"], "name": c["name"], "short": c.get("short", c["name"]), "summary": c["summary"],
                "score": round(match, 3), "percent": round(100 * match), "urgent": bool(c.get("urgent")),
                "sources": c["sources"],
                "matched": [self.symptoms[s]["label"] for s in prof if s in P],
                "against": [self.symptoms[s]["label"] for s in prof if s in A]
                           + [self.symptoms[s]["label"] for s in against if s in P],
                "unexplained": [self.symptoms[s]["label"] for s in real if s not in prof],
            })
        return sorted(out, key=lambda m: m["score"], reverse=True)

    # -------------------------------------------------------------- questions
    def _likelihood(self, cond, sid):
        if sid in cond["profile"]:
            return 0.08 + 0.8 * cond["profile"][sid] / 3
        if sid in cond.get("against", {}):
            return 0.03
        return 0.08

    def next_questions(self, state, matches, n=QUESTIONS_PER_TURN):
        """Unasked symptoms with the highest expected information gain over the top candidates."""
        known = set(state["present"]) | set(state["absent"]) | set(state["asked"])
        top = matches[:5]
        if not top:
            return [s for s in ("painful_periods", "heavy_bleeding", "irregular_cycles") if s not in known][:n]
        conds = [self.by_id[m["id"]] for m in top]
        w = [math.exp(6 * m["score"]) for m in top]
        p = [x / sum(w) for x in w]
        H = lambda q: -sum(x * math.log(x) for x in q if x > 0)
        h0 = H(p)
        gains = []
        for sid in self.symptoms:
            if sid in known:
                continue
            L = [self._likelihood(c, sid) for c in conds]
            if max(L) <= 0.08:
                continue
            # Personal context questions (age, pregnancy, tampon use) only when they really matter to a top candidate.
            if self.symptoms[sid]["kind"] == "context" and not any(
                    c["profile"].get(sid, 0) >= 3 or sid in c.get("against", {}) for c in conds[:3]):
                continue
            py = sum(pi * li for pi, li in zip(p, L))
            post_y = [pi * li / py for pi, li in zip(p, L)]
            post_n = [pi * (1 - li) / (1 - py) for pi, li in zip(p, L)]
            gain = h0 - py * H(post_y) - (1 - py) * H(post_n)
            if sid in conds[0]["profile"]:          # confirming the leader is useful even when it is alone
                gain += 0.03 * conds[0]["profile"][sid] / 3
            if sid in conds[0].get("key", []):
                gain += 0.05
            gains.append((gain, sid))
        gains.sort(reverse=True)
        return [sid for g, sid in gains if g > 0.01][:n]

    def questions_payload(self, ids):
        return [{"id": s, "label": self.symptoms[s]["label"], "text": self.symptoms[s]["question"]} for s in ids]

    # ---------------------------------------------------------------- safety
    def urgent(self, state):
        P = set(state["present"])
        return [f["message"] for f in self.red_flags if all(s in P for s in f["if_all"])]

    def is_ready(self, state, matches, questions, intent):
        if not matches:
            return False
        if intent == "results" or not questions or state["rounds"] >= MAX_ROUNDS:
            return True
        if matches[0]["urgent"] and matches[0]["score"] >= READY_SCORE:
            return True
        lead = matches[0]["score"] - (matches[1]["score"] if len(matches) > 1 else 0)
        return state["rounds"] >= 2 and matches[0]["score"] >= READY_SCORE and lead >= READY_MARGIN

    def picture(self, state):
        lab = lambda ids: [{"id": s, "label": self.symptoms[s]["label"]} for s in ids]
        return {"present": lab(state["present"]), "absent": lab(state["absent"])}

    def completeness(self, state, matches):
        """0-1: how much of the leading condition's profile Luna has checked (drives the moon)."""
        if not matches:
            return 0.0
        prof = self.by_id[matches[0]["id"]]["profile"]
        known = set(state["present"]) | set(state["absent"])
        return round(sum(w for s, w in prof.items() if s in known) / sum(prof.values()), 2)

    def vocabulary(self):
        return "\n".join(f"{s['id']}: {s['label']}" for s in self.symptoms.values())
