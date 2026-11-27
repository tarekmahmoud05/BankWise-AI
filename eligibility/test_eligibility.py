"""pytest suite for the eligibility engine.
Run from the folder that contains the rules files:  pytest -q"""
import math
from pathlib import Path

import pytest

from eligibility import check_eligibility, load_rules, monthly_payment, max_affordable_loan

HERE = Path(__file__).parent
RULES_DIR = HERE / "rules" 
PERSONAL = load_rules(RULES_DIR / "personal_loan_eligibility_rules_ar.json")
CAR = load_rules(RULES_DIR / "car_loan_eligibility_rules_ar.json")


def independent_payment(p, annual_pct, n):
    """Textbook annuity formula, written differently from the engine."""
    i = annual_pct / 100 / 12
    return round(p * i / (1 - (1 + i) ** -n), 2)


def base(**kw):
    a = {"employment_type": "private", "age": 30, "loan_amount": 200000, "tenor_months": 60,
         "employer_undertaking": "with_pledge", "net_monthly_income": 30000,
         "existing_monthly_installments": 0}
    a.update(kw)
    return a


# ---------------------------------------------------------------- math
@pytest.mark.parametrize("p,rate,n", [(100000, 20, 12), (200000, 25, 60), (50000, 18, 36), (750000, 23, 120)])
def test_payment_matches_independent_formula(p, rate, n):
    assert monthly_payment(p, rate, n) == independent_payment(p, rate, n)


def test_known_values():
    assert monthly_payment(100000, 20, 12) == 9263.45
    assert monthly_payment(200000, 25, 60) == 5870.26


def test_zero_rate_does_not_divide_by_zero():
    assert monthly_payment(12000, 0, 12) == 1000.0


def test_invalid_payment_inputs():
    with pytest.raises(ValueError):
        monthly_payment(-1, 20, 12)
    with pytest.raises(ValueError):
        monthly_payment(1000, 20, 0)


def test_max_affordable_loan_roundtrip():
    loan = max_affordable_loan(0.5, 10000, 0, 21.75, 60)
    assert monthly_payment(loan, 21.75, 60) == pytest.approx(5000, abs=0.5)


# ---------------------------------------------------------------- personal loan decisions
def test_private_employee_with_pledge_is_eligible():
    r = check_eligibility(PERSONAL, base())
    assert r["decision"] == "LIKELY_ELIGIBLE"
    assert r["program"]["id"] == "PL09"
    assert r["monthly_payment"] == independent_payment(200000, 21.75, 60)
    assert r["annual_rate"] == 21.75
    assert r["documents"] and "ختام" not in r["message"]
    assert r["total_interest"] == pytest.approx(r["monthly_payment"] * 60 - 200000, abs=0.01)


def test_message_has_no_unfilled_placeholders():
    r = check_eligibility(PERSONAL, base())
    assert "{" not in r["message"]


def test_debt_burden_failure_gives_max_affordable_loan():
    r = check_eligibility(PERSONAL, base(net_monthly_income=8000, loan_amount=300000))
    assert r["decision"] == "NOT_ELIGIBLE"
    codes = [f["reason_code"] for f in r["failed_rules"]]
    assert "DEBT_BURDEN" in codes
    msg = next(f for f in r["failed_rules"] if f["reason_code"] == "DEBT_BURDEN")
    assert "{" not in msg["suggestion"]


def test_dbr_boundary_exactly_50_percent_passes():
    pay = monthly_payment(200000, 21.75, 60)
    r = check_eligibility(PERSONAL, base(net_monthly_income=pay * 2))
    assert r["decision"] == "LIKELY_ELIGIBLE"


def test_existing_installments_count_toward_dbr():
    pay = monthly_payment(200000, 21.75, 60)
    r = check_eligibility(PERSONAL, base(net_monthly_income=pay * 2, existing_monthly_installments=100))
    assert r["decision"] == "NOT_ELIGIBLE"


def test_age_at_maturity_failure_and_suggestion():
    r = check_eligibility(PERSONAL, base(employment_type="government_public", age=58, loan_amount=100000,
                                         tenor_months=120, employer_contracted=True))
    assert r["decision"] == "NOT_ELIGIBLE"
    f = next(x for x in r["failed_rules"] if x["reason_code"] == "AGE_AT_MATURITY")
    assert "84" in f["suggestion"]          # (65 - 58) * 12


def test_age_at_maturity_boundary_passes():
    r = check_eligibility(PERSONAL, base(employment_type="government_public", age=58, loan_amount=100000,
                                         tenor_months=84, employer_contracted=True))
    assert r["decision"] == "LIKELY_ELIGIBLE"


def test_doctor_minimum_age_is_25():
    r = check_eligibility(PERSONAL, base(employment_type="doctor", age=23, loan_amount=100000))
    assert r["decision"] == "NOT_ELIGIBLE"
    assert any(f["reason_code"] == "MIN_AGE" for f in r["failed_rules"])


def test_loan_amount_above_program_limit():
    r = check_eligibility(PERSONAL, base(employment_type="doctor", age=40, loan_amount=8000000))
    assert r["decision"] == "NOT_ELIGIBLE"
    assert any(f["reason_code"] == "LOAN_AMOUNT_RANGE" for f in r["failed_rules"])


def test_tenor_below_program_minimum():
    r = check_eligibility(PERSONAL, base(tenor_months=6))      # PL09 minimum is 12 months
    assert r["decision"] == "NOT_ELIGIBLE"
    assert any(f["reason_code"] == "TENOR_RANGE" for f in r["failed_rules"])


def test_pensioner_age_limit_is_70():
    ok = check_eligibility(PERSONAL, base(employment_type="pensioner", age=68, loan_amount=100000, tenor_months=24))
    bad = check_eligibility(PERSONAL, base(employment_type="pensioner", age=68, loan_amount=100000, tenor_months=36))
    assert ok["decision"] == "LIKELY_ELIGIBLE"
    assert bad["decision"] == "NOT_ELIGIBLE"


def test_pension_program_depends_on_holder_and_source():
    r = check_eligibility(PERSONAL, base(employment_type="pensioner", age=60, loan_amount=100000,
                                         tenor_months=60, pension_holder="beneficiary", pension_armed_forces=True))
    assert r["program"]["id"] == "PL13"


def test_reserve_officer_needs_discharge_months_then_checks_them():
    a = base(employment_type="reserve_officer", loan_amount=50000, tenor_months=24, salary_transferred_to_bm=True)
    assert check_eligibility(PERSONAL, a)["decision"] == "NEED_MORE_INFO"
    ok = check_eligibility(PERSONAL, {**a, "months_until_discharge": 30})
    bad = check_eligibility(PERSONAL, {**a, "months_until_discharge": 12})
    assert ok["decision"] == "LIKELY_ELIGIBLE"
    assert any(f["reason_code"] == "TENOR_UNTIL_DISCHARGE" for f in bad["failed_rules"])


def test_flash_loan_requires_salary_transfer():
    a = base(employment_type="government_public", employer_undertaking="none", loan_amount=100000)
    r = check_eligibility(PERSONAL, {**a, "salary_transferred_to_bm": True, "employer_contracted": False})
    ids = {r["program"]["id"], *[p["id"] for p in r.get("other_eligible_programs", [])]}
    assert "PL01" in ids
    r2 = check_eligibility(PERSONAL, {**a, "salary_transferred_to_bm": False})
    ids2 = {r2["program"]["id"], *[p["id"] for p in r2.get("other_eligible_programs", [])]}
    assert "PL01" not in ids2


def test_cheapest_installment_is_chosen_when_several_programs_pass():
    a = base(employment_type="government_public", employer_undertaking="none", loan_amount=100000,
             employer_contracted=True, salary_transferred_to_bm=True)
    r = check_eligibility(PERSONAL, a)
    assert r["decision"] == "LIKELY_ELIGIBLE"
    pays = [r["monthly_payment"]] + [p["monthly_payment"] for p in r["other_eligible_programs"]]
    assert r["monthly_payment"] == min(pays)


def test_deemed_income_program_needs_no_income_inputs():
    r = check_eligibility(PERSONAL, {"employment_type": "judicial", "age": 45, "loan_amount": 200000,
                                     "tenor_months": 60})
    assert r["decision"] == "LIKELY_ELIGIBLE"
    assert r["program"]["id"] == "PL17"


def test_missing_required_fields():
    r = check_eligibility(PERSONAL, {"employment_type": "private", "age": 30})
    assert r["decision"] == "NEED_MORE_INFO"
    assert set(r["missing_fields"]) == {"loan_amount", "tenor_months"}


def test_missing_income_asks_only_for_income():
    r = check_eligibility(PERSONAL, {"employment_type": "private", "age": 30, "loan_amount": 100000,
                                     "tenor_months": 60, "employer_undertaking": "with_pledge"})
    assert r["decision"] == "NEED_MORE_INFO"
    assert set(r["missing_fields"]) == {"net_monthly_income", "existing_monthly_installments"}


@pytest.mark.parametrize("bad", [
    {"age": -5}, {"age": 5}, {"age": "abc"}, {"loan_amount": 0}, {"loan_amount": -1000},
    {"tenor_months": 0}, {"tenor_months": 12.5}, {"employment_type": "astronaut"},
    {"net_monthly_income": -1}, {"existing_monthly_installments": -1},
])
def test_invalid_inputs_never_crash(bad):
    r = check_eligibility(PERSONAL, base(**bad))
    assert r["decision"] == "INVALID_INPUT"
    assert r["errors"]


def test_arabic_digits_and_numeric_strings_are_accepted():
    r = check_eligibility(PERSONAL, base(age="٣٠", loan_amount="200,000", tenor_months="٦٠"))
    assert r["decision"] == "LIKELY_ELIGIBLE"


def test_unusual_inputs_do_not_raise():
    for weird in ({}, {"employment_type": ""}, {"age": None}, {"age": True}, {"loan_amount": float("inf")}):
        r = check_eligibility(PERSONAL, weird)
        assert r["decision"] in {"NEED_MORE_INFO", "INVALID_INPUT", "NOT_ELIGIBLE", "LIKELY_ELIGIBLE"}


def test_disclaimer_always_present():
    for a in (base(), base(age=-1), {}):
        assert check_eligibility(PERSONAL, a)["disclaimer"]


def test_disabled_assumption_rules_are_skipped_and_can_be_enabled():
    low_income = base(net_monthly_income=3000, loan_amount=20000, tenor_months=60)
    assert check_eligibility(PERSONAL, low_income)["decision"] == "LIKELY_ELIGIBLE"
    import copy
    rules = copy.deepcopy(PERSONAL)
    next(r for r in rules["rules"] if r["id"] == "R7")["enabled"] = True
    r = check_eligibility(rules, low_income)
    assert r["decision"] == "NOT_ELIGIBLE"
    assert any(f["reason_code"] == "MIN_INCOME" for f in r["failed_rules"])

def test_government_with_salary_transfer_does_not_crash():
    r = check_eligibility(PERSONAL, base(employment_type="government_public",
            employer_undertaking="none", salary_transferred_to_bm=True))
    assert r["decision"] in {"LIKELY_ELIGIBLE", "NOT_ELIGIBLE", "NEED_MORE_INFO"}


# ---------------------------------------------------------------- car loan file (team's rules)
def car(**kw):
    a = {"nationality": "egyptian", "age": 35, "employment_type": "private_sector", "car_condition": "new",
         "car_price": 800000, "loan_amount": 400000, "tenor_months": 60}
    a.update(kw)
    return a


def test_car_deemed_income_program_passes_without_income():
    r = check_eligibility(CAR, car())
    assert r["decision"] == "LIKELY_ELIGIBLE"
    assert r["monthly_payment"] == independent_payment(400000, r["annual_rate"], 60)


def test_car_non_egyptian_is_rejected():
    r = check_eligibility(CAR, car(nationality="other"))
    assert r["decision"] == "NOT_ELIGIBLE"
    assert r["failed_rules"][0]["reason_code"] == "NATIONALITY"


def test_car_underage_is_rejected():
    r = check_eligibility(CAR, car(age=19))
    assert r["decision"] == "NOT_ELIGIBLE"


def test_car_financing_ratio_too_high_for_deemed_asks_for_income_or_fails():
    r = check_eligibility(CAR, car(loan_amount=790000))
    assert r["decision"] in {"NEED_MORE_INFO", "NOT_ELIGIBLE", "LIKELY_ELIGIBLE"}
    assert "{" not in r["message"]


# ---------------------------------------------------------------- rules-file integrity
@pytest.mark.parametrize("rules", [PERSONAL, CAR], ids=["personal", "car"])
def test_rules_file_integrity(rules):
    ids = [p["id"] for p in rules["programs"]]
    assert len(ids) == len(set(ids))
    rule_ids = {r["id"] for r in rules["rules"]}
    legend = set(rules["metadata"]["source_legend"])
    for r in rules["rules"]:
        assert r["source"] in legend or "+" in r["source"]
    for p in rules["programs"]:
        assert p["min_loan_egp"] < p["max_loan_egp"]
        assert p["min_tenor_months"] < p["max_tenor_months"]
        assert p["annual_interest_rate_pct"] > 0
        assert set(p["rules_evaluated"]) <= rule_ids
        assert p["source"] in legend or "+" in p["source"]
