"""BankWise AI - واجهة تجريبية بسيطة (Streamlit).

مكانها: version2/app.py        (جنب requirements.txt)
التشغيل:   cd version2
           streamlit run app.py

بتستخدم نفس الـ agent بتاع الشات في التيرمنال مباشرة، فمحتاجة نفس إعدادات الـ .env.
"""
import os
import sys
import uuid

# chromadb محتاجة SQLite 3.35+ وسيرفرات Streamlit Cloud ممكن يبقى عليها نسخة أقدم.
# لو pysqlite3-binary متسطبة (على لينكس بس) بنستخدمها بدل sqlite3 العادية. لو مش موجودة بنكمّل عادي.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

# على Streamlit Cloud مفيش ملف .env: المفاتيح بتتحط في Settings > Secrets، وهنا بننقلها لـ environment
# variables قبل ما config.py يتقرا. محليًا (من غير secrets) الكود ده مابيعملش حاجة وبيستخدم .env.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from chatbot import config

st.set_page_config(page_title="BankWise AI", page_icon="🏦", layout="centered")

# كل فقرة تتحدد اتجاهها (عربي/إنجليزي) من أول حرف فيها، عشان الردود المختلطة تبان صح.
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li { unicode-bidi: plaintext; text-align: start; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "إيه المستندات المطلوبة لقرض السيارة؟",
    "ازاي اكلم خدمة العملاء؟",
    "عايز اعمل قرض شخصي",
    "لو اخدت قرض شخصي 100 ألف على 48 شهر القسط هيبقى كام؟",
    "i want a car loan, what should i do?",
]


@st.cache_resource(show_spinner="بحمّل الموديل... أول مرة بتاخد شوية وقت")
def get_agent():
    # بيتبني مرة واحدة بس ويتشارك بين كل الجلسات (الموديل + قاعدة البيانات + الـ LLM)
    from chatbot.cli import build_agent
    return build_agent()


def ask(question: str):
    st.session_state.pending = question


def show_debug(meta: dict):
    st.caption(
        f"tools: {meta['tools'] or '—'}  |  abstained: {meta['abstained']}  |  "
        f"reason: {meta['reason'] or '—'}  |  top_score: {meta['top_score']:.2f}  |  "
        f"{meta['latency_ms']} ms  |  tokens: {meta['tokens']}"
    )
    if meta["sources"]:
        with st.expander("المصادر المستخدمة"):
            for s in meta["sources"]:
                st.write(f"- {s['label']}")


# ------------------------------------------------------------------ state
if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role", "content", "meta"?}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("BankWise AI")
    st.caption(f"provider: {config.LLM_PROVIDER}  \nmodel: {config.LLM_MODEL or config.DEFAULT_MODELS.get(config.LLM_PROVIDER, '')}")
    debug = st.toggle("إظهار تفاصيل الـ debug", value=True)
    if st.button("محادثة جديدة", use_container_width=True):
        try:
            get_agent().reset(st.session_state.sid)
        except Exception:
            pass
        st.session_state.sid = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.write("أسئلة جاهزة للتجربة")
    for i, q in enumerate(EXAMPLES):
        st.button(q, key=f"ex{i}", use_container_width=True, on_click=ask, args=(q,))

# ------------------------------------------------------------------ main
st.title("🏦 BankWise AI")
st.caption("مساعد بنك مصر الافتراضي - نسخة تجريبية")

try:
    agent = get_agent()
except Exception as exc:                # مفتاح ناقص، موديل غلط، إلخ
    st.error(f"مقدرتش أشغّل الـ agent: {exc}")
    st.info("اتأكدي من ملف .env جوه فولدر version2 (LLM_PROVIDER و GEMINI_API_KEY و LLM_MODEL).")
    st.stop()

# السؤال: يا إما اتكتب في الشات، يا إما اتضغط عليه (سؤال جاهز / سؤال مقترح)
prompt = st.chat_input("اكتب سؤالك هنا...") or st.session_state.pop("pending", None)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if debug and m.get("meta"):
            show_debug(m["meta"])

if not st.session_state.messages:
    st.info("اكتبي سؤالك تحت، أو اختاري سؤال جاهز من القايمة الجانبية.")

# الأسئلة المقترحة: تحت آخر رد بس، وتختفي أول ما يتبعت سؤال جديد
last = st.session_state.messages[-1] if st.session_state.messages else None
if not prompt and last and last["role"] == "assistant" and last.get("meta", {}).get("suggestions"):
    st.caption("أسئلة ممكن تفيدك:")
    for i, q in enumerate(last["meta"]["suggestions"]):
        st.button(q, key=f"sg{len(st.session_state.messages)}_{i}", use_container_width=True,
                  on_click=ask, args=(q,))

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("بفكر..."):
            r = agent.chat(st.session_state.sid, prompt)

    meta = {
        "tools": r.tools_used, "abstained": r.abstained, "reason": r.reason,
        "top_score": r.top_score, "latency_ms": r.latency_ms,
        "tokens": f"{r.input_tokens}/{r.output_tokens}", "sources": r.sources,
        "suggestions": getattr(r, "suggestions", []),
    }
    st.session_state.messages.append({"role": "assistant", "content": r.text, "meta": meta})
    st.rerun()      # يعيد الرسم عشان الرد الجديد وأسئلته المقترحة يظهروا مع باقي المحادثة