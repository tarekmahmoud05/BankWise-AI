"""Central configuration. Everything tunable lives here or in environment variables (.env)."""
import os
import sys
from pathlib import Path

try:                                   # .env is optional
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ----------------------------------------------------------------- paths
VERSION2 = Path(__file__).resolve().parent.parent            # .../version2
EMBEDDING_DIR = VERSION2 / "embedding"
ELIGIBILITY_DIR = VERSION2 / "eligibility"
CHROMA_DIR = Path(os.getenv("BANKWISE_CHROMA_DIR", EMBEDDING_DIR / "chroma_db"))
COLLECTION = os.getenv("BANKWISE_COLLECTION", "bm_faq")

# The existing modules (search_chunks.py, tools.py, eligibility.py ...) are plain scripts, not a
# package, so we put their folders on sys.path instead of modifying them.
for _p in (EMBEDDING_DIR, ELIGIBILITY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ----------------------------------------------------------------- retrieval
TOP_K = int(os.getenv("BANKWISE_TOP_K", "4"))
# Cosine-similarity threshold. Below it we say "I don't know" instead of guessing.
# 0.50 is a STARTING value: calibrate it with embedding/eval_real/eval_retrieval.py (see the guide).
MIN_SCORE = float(os.getenv("BANKWISE_MIN_SCORE", "0.50"))

# ----------------------------------------------------------------- LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")        # anthropic | openai | gemini
LLM_MODEL = os.getenv("LLM_MODEL", "")                       # empty -> provider default below
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))
# Optional. Leave empty to use the provider default (some newer models restrict sampling parameters).
# For a bank bot a low value such as 0.2 is a good choice if your model accepts it.
LLM_TEMPERATURE = float(os.environ["LLM_TEMPERATURE"]) if os.getenv("LLM_TEMPERATURE") else None
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))

# ----------------------------------------------------------------- agent limits
MAX_TOOL_STEPS = 5            # max LLM<->tool round trips per user message (prevents loops)
MAX_USER_CHARS = 1000         # longer input is truncated
MAX_HISTORY_MESSAGES = 12     # last N messages kept per session (6 turns)

# ----------------------------------------------------------------- fixed texts (never written by the LLM)
# 19888 is the call-center number from the bank's own FAQ (entry CON-001).
FALLBACK_AR = ("عذراً، لا أملك معلومات مؤكدة للإجابة عن هذا السؤال. "
               "يمكنك التواصل مع مركز اتصالات بنك مصر على 19888 أو زيارة أقرب فرع.")
ERROR_AR = "حدث خطأ مؤقت. من فضلك حاول مرة أخرى بعد قليل، أو تواصل مع بنك مصر على 19888."
EMPTY_AR = "من فضلك اكتب سؤالك وسأساعدك."
