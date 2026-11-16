"""pytest for calculator.py and tools.py. Run from the eligibility folder:  pytest -q"""
from pathlib import Path

import pytest

from eligibility import load_rules, check_eligibility
from calculator import calculate_loan
from tools import dispatch, tool_definitions, openai_tools

HERE = Path(__file__).parent
RULES_DIR = HERE / "rules"  
PERSONAL = load_rules(RULES_DIR / "personal_loan_eligibility_rules_ar.json")
CAR = load_rules(RULES_DIR / "car_loan_eligibility_rules_ar.json")

PERSONAL_ANSWERS = {"employment_type": "private", "age": 30, "loan_amount": 200000,
                    "tenor_months": 60, "employer_undertaking": "with_pledge",
                    "net_monthly_income": 30000, "existing_monthly_installments": 0}
CAR_ANSWERS = {"nationality": "egyptian", "age": 35, "employment_type": "private_sector",
               "car_condition": "new", "car_price": 800000, "loan_amount": 400000,
               "tenor_months": 60}


# ---------------------------------------------------------------- calculator vs engine
def test_personal_matches_engine():
    r = check_eligibility(PERSONAL, PERSONAL_ANSWERS)
    c = calculate_loan(PERSONAL, 200000, 60, program_id=r["program"]["id"])
    o = c["options"][0]
    assert o["monthly_payment"] == r["monthly_payment"]
    assert o["total_interest"] == r["total_interest"]
    assert o["equivalent_flat_rate_pct"] == r["equivalent_flat_rate_pct"]


def test_car_matches_engine():
    r = check_eligibility(CAR, CAR_ANSWERS)
    c = calculate_loan(CAR, 400000, 60, program_id=r["program"]["id"], car_price=800000)
    assert c["options"][0]["monthly_payment"] == r["monthly_payment"]


# ---------------------------------------------------------------- filters
def test_out_of_range_is_skipped():
    c = calculate_loan(PERSONAL, 200000, 6, program_id="PL09")      # PL09 minimum is 12 months
    assert c["options"] == []
    assert c["out_of_range_programs"] == ["PL09"]


def test_options_sorted_cheapest_first():
    c = calculate_loan(PERSONAL, 200000, 60, employment_type="private")
    pays = [o["monthly_payment"] for o in c["options"]]
    assert len(pays) > 1
    assert pays == sorted(pays)


def test_car_financing_ratio_filters_programs():
    c = calculate_loan(CAR, 400000, 60, employment_type="private_sector",
                       car_price=800000, car_condition="new")
    ids = {o["program_id"] for o in c["options"]}
    assert "P2" in ids                       # 50% falls in the 40-50% band
    assert "P1" not in ids and "P1" in c["financing_ratio_mismatch"]
    assert "P3" not in ids                   # P3 is for used cars


def test_car_condition_used_only_used_programs():
    c = calculate_loan(CAR, 400000, 60, car_price=800000, car_condition="used")
    assert {o["program_id"] for o in c["options"]} == {"P3"}


# ---------------------------------------------------------------- inputs
@pytest.mark.parametrize("amount,months", [(-5, 12), (0, 12), (1000, 0), ("abc", 12), (1000, 12.5), (None, 12)])
def test_invalid_inputs_return_error(amount, months):
    c = calculate_loan(PERSONAL, amount, months)
    assert c["error"] and c["options"] == []


def test_arabic_digits_and_commas():
    c = calculate_loan(PERSONAL, "٢٠٠٬٠٠٠", "٦٠", program_id="PL09")
    assert c["options"][0]["monthly_payment"] == check_eligibility(PERSONAL, PERSONAL_ANSWERS)["monthly_payment"]


def test_always_marked_not_eligibility_and_has_disclaimer():
    c = calculate_loan(PERSONAL, 100000, 36)
    assert c["is_eligibility_check"] is False
    assert c["disclaimer"]


# ---------------------------------------------------------------- tools
def test_dispatch_personal_eligibility():
    r = dispatch("check_personal_loan_eligibility", PERSONAL_ANSWERS)
    assert r["decision"] == "LIKELY_ELIGIBLE"


def test_dispatch_car_eligibility():
    r = dispatch("check_car_loan_eligibility", CAR_ANSWERS)
    assert r["decision"] == "LIKELY_ELIGIBLE"


def test_dispatch_estimate():
    r = dispatch("estimate_payment", {"loan_type": "personal", "amount": 200000, "months": 60,
                                      "program_id": "PL09"})
    assert r["options"][0]["program_id"] == "PL09"


def test_dispatch_never_raises():
    assert "error" in dispatch("no_such_tool", {})
    assert "error" in dispatch("estimate_payment", {"loan_type": "boat", "amount": 1, "months": 1})
    assert "error" in dispatch("estimate_payment", {})


def test_schema_enums_match_rules_files():
    defs = {d["name"]: d for d in tool_definitions()}
    car_props = defs["check_car_loan_eligibility"]["parameters"]["properties"]
    personal_props = defs["check_personal_loan_eligibility"]["parameters"]["properties"]
    assert "private_sector" in car_props["employment_type"]["enum"]
    assert "private" in personal_props["employment_type"]["enum"]
    assert set(personal_props["employer_undertaking"]["enum"]) == {"with_pledge", "none"}
    assert "with_salary_transfer" in car_props["employer_undertaking"]["enum"]


def test_schema_required_fields_only_the_mandatory_ones():
    defs = {d["name"]: d for d in tool_definitions()}
    assert set(defs["check_personal_loan_eligibility"]["parameters"]["required"]) == {
        "employment_type", "age", "loan_amount", "tenor_months"}
    assert len(defs["check_car_loan_eligibility"]["parameters"]["required"]) == 7


def test_openai_wrapper_shape():
    t = openai_tools()
    assert all(x["type"] == "function" and "name" in x["function"] for x in t)
