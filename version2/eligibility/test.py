import json
from pathlib import Path
from eligibility import load_rules, check_eligibility

HERE = Path(__file__).parent
RULES_DIR = HERE / "rules" 
rules_per = load_rules(RULES_DIR / "personal_loan_eligibility_rules_ar.json")
rules_car = load_rules(RULES_DIR / "car_loan_eligibility_rules_ar.json")

answers = {
    "nationality": "other",
    "age": 60,
    "employment_type": "private_sector",
    "car_condition": "new",
    "car_price": 800000,
    "loan_amount": 400000,
    "tenor_months": 60,
}

print(json.dumps(check_eligibility(rules_car, answers), ensure_ascii=False, indent=2))