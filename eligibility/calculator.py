"""Standalone loan estimate (monthly payment, total, interest) for Banque Misr programs.

It is NOT an eligibility check. It reads the rate and limits from the rules JSON,
so the LLM never has to guess a rate. It reuses monthly_payment() from the engine,
so the numbers always match check_eligibility.
"""
import math

from eligibility import monthly_payment, ARABIC_DIGITS


def _num(value, integer=False):
    """Convert user/LLM input to a positive number. Raises ValueError if invalid."""
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, str):
        value = value.translate(ARABIC_DIGITS).strip().replace(",", "")
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise ValueError
    if integer:
        if n != int(n):
            raise ValueError
        return int(n)
    return n


def _result(rules, options, out_of_range, ratio_mismatch, error=None):
    out = {
        "options": options,
        "out_of_range_programs": out_of_range,
        "financing_ratio_mismatch": ratio_mismatch,
        "is_eligibility_check": False,
        "disclaimer": rules["metadata"]["disclaimer"],
    }
    if error:
        out["error"] = error
    return out


def calculate_loan(rules, amount, months, program_id=None, employment_type=None,
                   car_price=None, car_condition=None):
    """Return estimates for every program that fits the filters, cheapest installment first.

    program_id       : only this program.
    employment_type  : only programs for this work type (plus programs with no work-type condition).
    car_price        : (car loans) filters by financing ratio = amount / car_price.
    car_condition    : (car loans) "new" or "used".
    """
    try:
        amount = _num(amount)
        months = _num(months, integer=True)
        if car_price is not None:
            car_price = _num(car_price)
    except (TypeError, ValueError):
        return _result(rules, [], [], [], error="amount, months and car_price must be positive numbers")

    progs = rules["programs"]
    if program_id:
        progs = [p for p in progs if p["id"] == program_id]
    if employment_type:
        progs = [p for p in progs
                 if employment_type in p["applies_when"].get("employment_type", [employment_type])]
    if car_condition:
        progs = [p for p in progs
                 if car_condition in p["applies_when"].get("car_condition", [car_condition])]

    options, out_of_range, ratio_mismatch = [], [], []
    for p in progs:
        if not (p["min_loan_egp"] <= amount <= p["max_loan_egp"]
                and p["min_tenor_months"] <= months <= p["max_tenor_months"]):
            out_of_range.append(p["id"])
            continue
        fr = p.get("financing_ratio")
        if fr and car_price:
            ratio = amount / car_price
            if not (fr["min_exclusive"] < ratio <= fr["max_inclusive"] + 1e-9):
                ratio_mismatch.append(p["id"])
                continue
        pay = monthly_payment(amount, p["annual_interest_rate_pct"], months)
        total = round(pay * months, 2)
        interest = round(total - amount, 2)
        options.append({
            "program_id": p["id"],
            "program_name": p["name"],
            "annual_rate": p["annual_interest_rate_pct"],
            "monthly_payment": pay,
            "total_repayment": total,
            "total_interest": interest,
            "equivalent_flat_rate_pct": round(interest / amount / (months / 12) * 100, 2),
        })
    options.sort(key=lambda o: o["monthly_payment"])
    return _result(rules, options, out_of_range, ratio_mismatch)
