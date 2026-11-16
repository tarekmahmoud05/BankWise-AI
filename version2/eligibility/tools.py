"""Tool definitions for the LLM + a dispatcher that runs them.

The eligibility schemas are generated from the `inputs` section of each rules JSON,
so enum values always match the files (no INVALID_INPUT from a wrong value).

Usage:
    from tools import openai_tools, tool_definitions, dispatch

    tools = openai_tools()                       # OpenAI format
    decls = tool_definitions()                   # plain {name, description, parameters}; use for Gemini
    result = dispatch(tool_name, tool_args, lang="ar")   # returns a JSON-serializable dict
"""
import json
from pathlib import Path

from eligibility import load_rules, check_eligibility
from calculator import calculate_loan

HERE = Path(__file__).parent
RULES_DIR = HERE / "rules"  
RULE_FILES = {
    "personal": "personal_loan_eligibility_rules",
    "car": "car_loan_eligibility_rules",
}

_TYPES = {"integer": "integer", "number": "number", "boolean": "boolean", "enum": "string"}
_cache = {}


# ---------------------------------------------------------------- loading
def get_rules(loan_type, lang="ar"):
    """Load (and cache) the rules file. Falls back to Arabic if the language file is missing."""
    base = RULE_FILES[loan_type]
    path = RULES_DIR / f"{base}_{lang}.json"
    if not path.exists():
        path = RULES_DIR / f"{base}_ar.json"
    key = str(path)
    if key not in _cache:
        _cache[key] = load_rules(path)
    return _cache[key]


# ---------------------------------------------------------------- schemas
def _eligibility_parameters(rules):
    props, required = {}, []
    for i in rules["inputs"]:
        p = {"type": _TYPES[i["type"]]}
        desc = i["label"]
        if i["type"] == "enum":
            p["enum"] = [v["id"] for v in i["values"]]
            desc += " | " + "; ".join(f"{v['id']} = {v['label']}" for v in i["values"])
        if i.get("ask_only_if"):
            desc += f" | ask only if: {i['ask_only_if']}"
        if "default" in i:
            desc += f" | default: {i['default']} (use it if the user does not know; do not guess)"
        if i.get("required") == "conditional":
            desc += " | do NOT ask up front; only if the tool returns NEED_MORE_INFO"
        p["description"] = desc
        props[i["id"]] = p
        if i.get("required") is True:
            required.append(i["id"])
    return {"type": "object", "properties": props, "required": required}


def tool_definitions(lang="ar"):
    personal = get_rules("personal", lang)
    car = get_rules("car", lang)
    return [
        {
            "name": "check_personal_loan_eligibility",
            "description": (
                "Pre-screen eligibility for a Banque Misr PERSONAL loan. Pass only values the user "
                "actually stated. If the result is NEED_MORE_INFO, ask the user for exactly "
                "missing_fields and call again. Never change the decision."
            ),
            "parameters": _eligibility_parameters(personal),
        },
        {
            "name": "check_car_loan_eligibility",
            "description": (
                "Pre-screen eligibility for a Banque Misr CAR loan. Pass only values the user "
                "actually stated. If the result is NEED_MORE_INFO, ask the user for exactly "
                "missing_fields and call again. Never change the decision."
            ),
            "parameters": _eligibility_parameters(car),
        },
        {
            "name": "estimate_payment",
            "description": (
                "Estimate monthly installment and total interest for a given amount and tenor. "
                "NOT an eligibility check. Never compute payments yourself. If the program is "
                "unknown, leave program_id empty and present the range (lowest to highest installment)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_type": {"type": "string", "enum": ["personal", "car"]},
                    "amount": {"type": "number", "description": "Loan amount in EGP"},
                    "months": {"type": "integer", "description": "Tenor in months"},
                    "program_id": {"type": "string", "description": "Optional, e.g. PL09 or P2"},
                    "employment_type": {"type": "string", "description": "Optional, an id from the rules file"},
                    "car_price": {"type": "number", "description": "Car loans only: car price in EGP"},
                    "car_condition": {"type": "string", "enum": ["new", "used"],
                                      "description": "Car loans only"},
                },
                "required": ["loan_type", "amount", "months"],
            },
        },
    ]


def openai_tools(lang="ar"):
    return [{"type": "function", "function": d} for d in tool_definitions(lang)]


# ---------------------------------------------------------------- dispatcher
def dispatch(name, args, lang="ar"):
    """Run a tool call coming from the LLM. Always returns a dict (never raises)."""
    try:
        args = dict(args or {})
        if name == "check_personal_loan_eligibility":
            return check_eligibility(get_rules("personal", lang), args)
        if name == "check_car_loan_eligibility":
            return check_eligibility(get_rules("car", lang), args)
        if name == "estimate_payment":
            rules = get_rules(args["loan_type"], lang)
            return calculate_loan(
                rules, args["amount"], args["months"],
                program_id=args.get("program_id") or None,
                employment_type=args.get("employment_type") or None,
                car_price=args.get("car_price"),
                car_condition=args.get("car_condition") or None,
            )
        return {"error": f"unknown tool: {name}"}
    except Exception as exc:  # keep the chat alive; the LLM can explain the failure
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(json.dumps(tool_definitions(), ensure_ascii=False, indent=2)[:3000])
