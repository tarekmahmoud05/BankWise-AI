"""Rule-based pre-screening for Banque Misr loans (personal and car).

The rules live in a JSON file (data). This module only executes them:
there is no eval(), every rule is a plain Python function chosen by its
`check` name, and the LLM never decides eligibility, it only explains the result.

Usage:
    rules = load_rules("personal_loan_eligibility_rules_ar.json")
    result = check_eligibility(rules, {"employment_type": "private", "age": 30,
                                       "loan_amount": 200000, "tenor_months": 60,
                                       "employer_undertaking": "with_pledge",
                                       "net_monthly_income": 15000,
                                       "existing_monthly_installments": 0})
    result["decision"]  # LIKELY_ELIGIBLE / NOT_ELIGIBLE / NEED_MORE_INFO / INVALID_INPUT
"""
import json
import math
import re
from pathlib import Path

# The car loan file (written by the team) has no `check` key, so map its rule ids.
CAR_RULE_CHECKS = {
    "R1": "nationality_egyptian", "R2": "min_age", "R3": "amount_range",
    "R4": "tenor_range", "R5": "financing_ratio", "R7": "max_age_at_maturity",
    "R8": "dbr", "R9": "min_income", "R10": "min_months",
}
INCOME_CHECKS = {"dbr", "min_income", "min_months"}
INCOME_INPUTS = {
    "dbr": ["net_monthly_income", "existing_monthly_installments"],
    "min_income": ["net_monthly_income"],
    "min_months": ["months_in_current_job_or_business"],
}
NON_NEGATIVE = {"existing_monthly_installments", "months_in_current_job_or_business"}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")


# ---------------------------------------------------------------- loan math
def payment_factor(annual_rate_pct, months):
    """Installment per 1 EGP of principal (reducing-balance / annuity)."""
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return 1 / months
    return r + r / ((1 + r) ** months - 1)


def monthly_payment(principal, annual_rate_pct, months):
    if principal <= 0 or months <= 0:
        raise ValueError("principal and months must be positive")
    return round(principal * payment_factor(annual_rate_pct, months), 2)


def max_affordable_loan(max_dbr, income, existing, annual_rate_pct, months):
    room = max_dbr * income - existing
    return max(0.0, room / payment_factor(annual_rate_pct, months))


# ---------------------------------------------------------------- loading
def load_rules(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _fmt(template, values):
    return template.format_map(_Safe(values))


def _money(x):
    return f"{x:,.0f}"


# ---------------------------------------------------------------- inputs
def _normalize(rules, answers):
    """Apply defaults, convert numeric strings, validate. Returns (answers, errors)."""
    spec = {i["id"]: i for i in rules["inputs"]}
    a, errors = {}, []
    for key, val in answers.items():
        if isinstance(val, str):
            val = val.translate(ARABIC_DIGITS).strip()
        a[key] = val
    for key, item in spec.items():
        if key not in a or a[key] in (None, ""):
            a.pop(key, None)
            if "default" in item:
                a[key] = item["default"]
            continue
        val, kind = a[key], item["type"]
        if kind in ("integer", "number"):
            try:
                if isinstance(val, bool):
                    raise ValueError
                num = float(str(val).replace(",", ""))
                if kind == "integer":
                    if num != int(num):
                        raise ValueError
                    num = int(num)
            except ValueError:
                errors.append(f"{item['label']}: يجب أن تكون رقماً")
                continue
            low_ok = num >= 0 if key in NON_NEGATIVE else num > 0
            if not low_ok:
                errors.append(f"{item['label']}: قيمة غير صالحة")
            if key == "age" and not 18 <= num <= 100:
                errors.append(f"{item['label']}: يجب أن يكون بين 18 و100")
            a[key] = num
        elif kind == "boolean":
            if isinstance(val, str):
                val = {"true": True, "false": False, "نعم": True, "لا": False}.get(val.lower(), val)
            if not isinstance(val, bool):
                errors.append(f"{item['label']}: يجب أن تكون نعم أو لا")
            a[key] = val
        elif kind == "enum":
            if val not in [v["id"] for v in item["values"]]:
                errors.append(f"{item['label']}: اختيار غير معروف")
    return a, errors


# ---------------------------------------------------------------- rule checks
# Each check returns (passed, details). `details` feeds the message placeholders.
def _check_nationality(a, p, params, x):
    return a.get("nationality") == "egyptian", {}


def _check_min_age(a, p, params, x):   # 21
    lim = (p.get("age") or {}).get("min_age", params.get("min_age"))
    if lim is None:
        return True, {}
    return a["age"] >= lim, {"min_age": lim}


def _check_max_age(a, p, params, x): # 65
    lim = (p.get("age") or {}).get("max_age_at_maturity", params.get("max_age_at_maturity"))
    if lim is None:
        return True, {}
    months_at_end = a["age"] * 12 + a["tenor_months"]
    return months_at_end <= lim * 12, {
        "age_at_maturity": round(months_at_end / 12, 1),
        "max_age_at_maturity": lim,
        "max_tenor_for_age": max(0, math.floor((lim - a["age"]) * 12)),
    }


def _check_amount(a, p, params, x): # max 5 m 
    ok = p["min_loan_egp"] <= a["loan_amount"] <= p["max_loan_egp"]
    return ok, {"loan_amount": _money(a["loan_amount"]), "min_loan": _money(p["min_loan_egp"]),
                "max_loan": _money(p["max_loan_egp"])}


def _check_tenor(a, p, params, x): # max 92 
    ok = p["min_tenor_months"] <= a["tenor_months"] <= p["max_tenor_months"]
    return ok, {"tenor_months": a["tenor_months"], "min_tenor": p["min_tenor_months"],
                "max_tenor": p["max_tenor_months"]}


def _check_financing_ratio(a, p, params, x): ## ?? need to check 
    fr = p.get("financing_ratio")
    if not fr:
        return True, {}
    ratio = a["loan_amount"] / a["car_price"]
    ok = fr["min_exclusive"] < ratio <= fr["max_inclusive"] + 1e-9
    return ok, {"ratio_pct": round(ratio * 100, 1), "max_ratio_pct": round(fr["max_inclusive"] * 100)}


def _check_dbr(a, p, params, x):  # max 40% 
    max_dbr = params["max_dbr"]
    income = a["net_monthly_income"]
    dbr = (a["existing_monthly_installments"] + x["monthly_payment"]) / income
    afford = max_affordable_loan(max_dbr, income, a["existing_monthly_installments"],
                                 p["annual_interest_rate_pct"], a["tenor_months"])
    afford = min(afford, p["max_loan_egp"])
    return dbr <= max_dbr + 1e-9, {"dbr": round(dbr, 4), "dbr_pct": round(dbr * 100, 1),
                                   "max_dbr_pct": round(max_dbr * 100),
                                   "max_affordable_loan": _money(math.floor(afford / 1000) * 1000)}


def _check_min_income(a, p, params, x):
    lim = params["min_net_monthly_income_egp"]
    return a["net_monthly_income"] >= lim, {"min_income": _money(lim)}


def _check_min_months(a, p, params, x):
    lim = params["min_months"]
    return a["months_in_current_job_or_business"] >= lim, {"min_months": lim}


def _check_discharge(a, p, params, x):
    left = a["months_until_discharge"]
    return a["tenor_months"] <= left, {"months_until_discharge": left, "tenor_months": a["tenor_months"]}


CHECKS = {
    "nationality_egyptian": _check_nationality, "min_age": _check_min_age,
    "max_age_at_maturity": _check_max_age, "amount_range": _check_amount,
    "tenor_range": _check_tenor, "financing_ratio": _check_financing_ratio,
    "dbr": _check_dbr, "min_income": _check_min_income, "min_months": _check_min_months,
    "tenor_until_discharge": _check_discharge,
}


def _rule_check_name(rule):
    return rule.get("check") or CAR_RULE_CHECKS.get(rule["id"])


def _enabled(rule):
    return rule.get("enabled", True)


# ---------------------------------------------------------------- program matching
def _matches(program, a):
    for key, allowed in program["applies_when"].items():
        if not isinstance(allowed, (list, tuple, set)):
            allowed = [allowed]
        if a.get(key) not in allowed:
            return False
    return True


def _missing_inputs(rules_by_id, program, a):
    needed = list(program.get("extra_required_inputs", []))
    for rid in program["rules_evaluated"]:
        rule = rules_by_id.get(rid)
        if not rule or not _enabled(rule):
            continue
        needed += INCOME_INPUTS.get(_rule_check_name(rule), [])
    seen, missing = set(), []
    for key in needed:
        if key not in a and key not in seen:
            seen.add(key)
            missing.append(key)
    return missing


def _evaluate(rules_by_id, program, a, skip_global=True):
    """Run the program's rules. Returns (payment, results, missing)."""
    missing = _missing_inputs(rules_by_id, program, a)
    pay = monthly_payment(a["loan_amount"], program["annual_interest_rate_pct"], a["tenor_months"])
    x = {"monthly_payment": pay}
    results = []
    for rid in program["rules_evaluated"]:
        rule = rules_by_id.get(rid)
        if rule is None or not _enabled(rule) or (skip_global and rule.get("scope") == "global"):
            continue
        name = _rule_check_name(rule)
        if name in INCOME_CHECKS and any(k in missing for k in INCOME_INPUTS[name]):
            continue
        if name == "tenor_until_discharge" and ("months_until_discharge" in missing or "months_until_discharge" not in a):
            continue
        ok, details = CHECKS[name](a, program, rule.get("parameters", {}), x)
        item = {"rule_id": rid, "name": rule["name"], "passed": ok, "details": details}
        if not ok:
            f = rule["on_fail"]
            values = {**details, **a}
            item.update(reason_code=f["reason_code"], message=_fmt(f["message"], values),
                        suggestion=_fmt(f["suggestion"], values), source=rule.get("source"),
                        indicative=rule.get("indicative", False))
        results.append(item)
    return pay, results, missing


# ---------------------------------------------------------------- documents
def _when_matches(when, program, a):
    pairs = re.findall(r'(?:program\.)?(\w+)\s*==\s*"([^"]+)"', when)
    return bool(pairs) and all(
        (program.get(k) if k in program else a.get(k)) == v for k, v in pairs)


def _documents(rules, program, a):
    if "documents" in program:
        return list(program["documents"])
    handoff = rules.get("branch_handoff", {})
    groups = list(program.get("document_groups", []))
    for cond in handoff.get("conditional_groups", []):
        if _when_matches(cond["when"], program, a):
            groups += cond["groups"]
    docs = []
    for g in groups:
        docs += handoff.get("document_groups", {}).get(g, {}).get("documents", [])
    return docs


# ---------------------------------------------------------------- main entry
def check_eligibility(rules, answers):
    meta = rules["metadata"]
    base = {"disclaimer": meta["disclaimer"], "rules_version": meta["version"]}
    decisions = rules["decisions"]
    a, errors = _normalize(rules, answers)
    if errors:
        return {**base, "decision": "INVALID_INPUT", "label": decisions.get("INVALID_INPUT", {}).get("label", ""),
                "errors": errors,
                "message": _fmt(decisions.get("INVALID_INPUT", {}).get("message_template", "{errors}"),
                                {"errors": "؛ ".join(errors)})}
    spec = {i["id"]: i for i in rules["inputs"]}
    required = [k for k, i in spec.items() if i["required"] is True and k not in a]
    if "months_until_discharge" in spec and a.get("employment_type") == "reserve_officer" \
            and "months_until_discharge" not in a:
        required.append("months_until_discharge")
    if required:
        return _need_more(base, rules, spec, required)

    rules_by_id = {r["id"]: r for r in rules["rules"]}
    # global rules (for example nationality in the car file)
    global_fail = []
    for r in rules["rules"]:
        if r.get("scope") == "global" and _enabled(r):
            ok, details = CHECKS[_rule_check_name(r)](a, {}, r.get("parameters", {}), {})
            if not ok:
                f = r["on_fail"]
                values = {**details, **a}
                global_fail.append({"rule_id": r["id"], "reason_code": f["reason_code"],
                                    "message": _fmt(f["message"], values),
                                    "suggestion": _fmt(f["suggestion"], values)})
    if global_fail:
        return _not_eligible(base, rules, None, global_fail)

    candidates = [p for p in rules["programs"] if _matches(p, a)]
    if not candidates:
        reason = {"rule_id": "R-select", "reason_code": "NO_MATCHING_PROGRAM",
                  "message": "لا يوجد برنامج تمويل مطابق لحالتك.",
                  "suggestion": "يمكن لأقرب فرع لبنك مصر إفادتك بالخيارات الأخرى."}
        return _not_eligible(base, rules, None, [reason])

    passing, pending, failing = [], [], []
    for p in candidates:
        pay, results, missing = _evaluate(rules_by_id, p, a)
        failed = [r for r in results if not r["passed"]]
        entry = {"program": p, "payment": pay, "results": results, "failed": failed, "missing": missing}
        if failed:
            failing.append(entry)
        elif missing:
            pending.append(entry)
        else:
            passing.append(entry)

    if passing:
        passing.sort(key=lambda e: (e["payment"], e["program"]["annual_interest_rate_pct"]))
        best = passing[0]
        p = best["program"]
        total = round(best["payment"] * a["tenor_months"], 2)
        total_interest = round(total - a["loan_amount"], 2)
        flat_rate = round(total_interest / a["loan_amount"] / (a["tenor_months"] / 12) * 100, 2)
        out = {**base, "decision": "LIKELY_ELIGIBLE", "label": decisions["LIKELY_ELIGIBLE"]["label"],
               "program": {"id": p["id"], "name": p["name"]},
               "monthly_payment": best["payment"], "annual_rate": p["annual_interest_rate_pct"],
               "equivalent_flat_rate_pct": flat_rate,
               "tenor_months": a["tenor_months"], "loan_amount": a["loan_amount"],
               "total_repayment": total, "total_interest": total_interest,
               "checks": best["results"], "documents": _documents(rules, p, a),
               "other_eligible_programs": [{"id": e["program"]["id"], "name": e["program"]["name"],
                                            "monthly_payment": e["payment"]} for e in passing[1:]],
               "indicative_assumptions": [r["rule_id"] for r in best["results"]
                                          if rules_by_id[r["rule_id"]].get("indicative")]}
        for r in best["results"]:
            if _rule_check_name(rules_by_id[r["rule_id"]]) == "dbr":
                out["dbr"] = r["details"].get("dbr")
        out["message"] = _fmt(decisions["LIKELY_ELIGIBLE"]["message_template"], {
            "program_name": p["name"], "monthly_payment": f"{best['payment']:,.2f}",
            "annual_rate": p["annual_interest_rate_pct"], "tenor_months": a["tenor_months"]})
        return out

    if pending:
        miss = []
        for e in pending:
            miss += [m for m in e["missing"] if m not in miss]
        return _need_more(base, rules, spec, miss)

    failing.sort(key=lambda e: (len(e["failed"]), e["program"]["annual_interest_rate_pct"]))
    best = failing[0]
    return _not_eligible(base, rules, best["program"], best["failed"], best["payment"])


def _need_more(base, rules, spec, fields):
    labels = [spec[f]["label"] if f in spec else f for f in fields]
    tpl = rules["decisions"]["NEED_MORE_INFO"]
    return {**base, "decision": "NEED_MORE_INFO", "label": tpl["label"], "missing_fields": fields,
            "missing_labels": labels,
            "message": _fmt(tpl["message_template"], {"missing_fields": "، ".join(labels)})}


def _not_eligible(base, rules, program, failed, payment=None):
    tpl = rules["decisions"]["NOT_ELIGIBLE"]
    first = failed[0]
    clean = [{k: v for k, v in f.items() if k in ("rule_id", "name", "reason_code", "message",
                                                   "suggestion", "source", "indicative")} for f in failed]
    out = {**base, "decision": "NOT_ELIGIBLE", "label": tpl["label"], "failed_rules": clean,
           "message": _fmt(tpl["message_template"], {"reason": first["message"],
                                                      "suggestion": first["suggestion"]})}
    if program:
        out["program"] = {"id": program["id"], "name": program["name"]}
        out["monthly_payment"] = payment
    return out
