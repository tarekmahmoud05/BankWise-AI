"""
check_loan_rates.py

بيسحب برامج القروض من API بنك مصر، ويقارن نسبة الفايدة بالقيمة اللي في ملفات
القواعد (eligibility/rules)، ولو اتغيرت بيعدلها في الملف بعد ما ياخد نسخة احتياطية.
حدود المبلغ والمدة بيبلّغ عنها في اللوج بس. وفيه كود جاهز لتعديلها، معطّل بعلامة #
(دور على "(1/2)" و"(2/2)" في الملف).

    python check_loan_rates.py --dry-run    يعرض الفروق بس، ومش بيعدل أي حاجة
    python check_loan_rates.py              يعرض الفروق ويعدل الملفات

مكان الملف:  data/api_data/check_loan_rates.py
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

# ============================== Config ==============================
BASE_URL = os.environ.get("BASE_URL", "https://www.banquemisr.com")
LANG = os.environ.get("LANG_CODE", "ar-EG")  

HERE = Path(__file__).resolve().parent
RULES_DIR = Path(os.environ.get("RULES_DIR", HERE.parents[1] / "eligibility" / "rules"))
BACKUP_DIR = HERE / "backups"         
LOG_FILE = HERE / "check_loan_rates.log"

CAR_MAPPING = {
    "الدخل الافتراضى  لقروض ومرابحات السيارات الجديدة  نسب التمويل 40%": "P1",
    "الدخل الافتراضى  لقروض ومرابحات السيارات الجديدة  نسب التمويل 50%": "P2",
    "الدخل الافتراضى  لقروض ومرابحات  السيارات المستعمله نسب التمويل 50%": "P3",
    "الدخل الافتراضى  لقروض ومرابحات السيارات الجديدة  نسب التمويل 60%": "P4",
    "الدخل الافتراضى  لقروض ومرابحات  السيارات المستعمله نسب التمويل 60%": "P5",
    "الموظفين - بموجب شهاده دخل": "P6",
    "قروض سيارات موظفين ق.ع اثبات دخل ج.متعاقدة": "P7",
    "الموظفين - تعهد جهه العمل بتحويل الراتب / القسط": "P8",
    "الموظفين بتعهد من جهة العمل-قطاع عام بدون تعهد تحويل مستحقات": "P9",
    "العملاء اصحاب المهن / الاعمال الحره بموجب اثبات دخل": "P10",
}

LOANS = {
    "personal_loan": {
        "type_id": os.environ.get("PERSONAL_LOAN_ID", "257a5aef-78ee-41ad-8b10-371d4de1c39b"),
        "rules_file": "personal_loan_eligibility_rules_ar.json",
        "extra_mapping": {},
    },
    "car_loan": {
        "type_id": os.environ.get("CAR_LOAN_ID", "d5de4d5e-5a9c-4c7c-975e-4304b1b68ac1"),
        "rules_file": "car_loan_eligibility_rules_ar.json",
        "extra_mapping": CAR_MAPPING,
    },
}


LIMIT_FIELDS = {
    "MinLoanInEGP": "min_loan_egp",
    "MaxLoanInEGP": "max_loan_egp",
    "MinDurationInMonths": "min_tenor_months",
    "MaxDurationInMonths": "max_tenor_months",
}
RATE_FIELD = "annual_interest_rate_pct"
# ====================================================================

for _s in (sys.stdout, sys.stderr):  
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("loan-check")


def norm(name: str) -> str:
    """بيوحد الشرطات والمسافات وكتابة (ى/ي، ة/ه، أ/إ/آ) عشان اختلاف بسيط في الاسم ما يبوظش الربط."""
    name = name.replace("\u2013", "-").replace("\u2014", "-")
    name = name.translate(str.maketrans("ىةأإآ", "يهااا"))
    return re.sub(r"\s+", " ", name).strip().lower()


def to_number(value) -> float:
    return float(str(value).replace(",", "").strip())


def fetch_programs(loan_type_id: str) -> list:
    url = f"{BASE_URL}/api/sitecore/LoanCalculator/GetLoanPrograms"
    params = {"loanTypeId": loan_type_id, "lang": LANG}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    last_error = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                raise ValueError("response is not a non-empty list")
            return data
        except Exception as e:
            last_error = e
            log.warning("attempt %d/3 failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(5 * attempt)
    raise last_error


def build_mapping(file_programs: dict, extra_mapping: dict) -> dict:
    """اسم البرنامج عند البنك -> id. من calculator_names في الملف + أي ربط إضافي في السكريبت."""
    mapping = {}
    for pid, prog in file_programs.items():
        for bank_name in prog.get("calculator_names", []):
            mapping[bank_name] = pid
    mapping.update(extra_mapping)
    return mapping


def analyze(api_items: list, mapping: dict, file_programs: dict):
    """
    بيقارن رد الـ API بالملف.
    بيرجع: (rate_changes, limit_changes, warnings)
      rate_changes:  {program_id: (old_rate, new_rate)}               <- بتتعدل أوتوماتيك
      limit_changes: {program_id: {file_key: (old, new)}}             <- بتتعدل بس لو فعّلتي (1/2) و(2/2)
      warnings:      رسائل للمراجعة اليدوية (برنامج جديد/اتشال/تضارب)
    """
    name_to_pid = {norm(k): v for k, v in mapping.items()}
    by_pid, warnings = defaultdict(list), []

    for item in api_items:
        pid = name_to_pid.get(norm(item.get("Name", "")))
        if pid is None:
            warnings.append(
                f"UNMAPPED bank program: {item.get('Name')!r} "
                f"(rate {item.get('AnnualInterestRate')}) - add its name to calculator_names in the rules file (car loan: CAR_MAPPING here)"
            )
        else:
            by_pid[pid].append(item)

    api_names = {norm(i.get("Name", "")) for i in api_items}
    for bank_name, pid in mapping.items():
        if norm(bank_name) not in api_names:
            warnings.append(f"{pid}: bank program {bank_name!r} not returned by the API anymore (removed or renamed?)")

    rate_changes, limit_changes = {}, {}
    for pid, items in by_pid.items():
        prog = file_programs.get(pid)
        if prog is None:
            warnings.append(f"{pid}: in the mapping but not found in the rules file")
            continue

        try:
            rates = {to_number(i["AnnualInterestRate"]) for i in items}
        except (KeyError, ValueError) as e:
            warnings.append(f"{pid}: unreadable AnnualInterestRate ({e}) - skipped")
            continue
        if len(rates) > 1:
            warnings.append(f"{pid}: bank entries disagree on the rate {sorted(rates)} - skipped")
        else:
            new, old = rates.pop(), prog[RATE_FIELD]
            if not 0 < new < 100:
                warnings.append(f"{pid}: implausible rate {new} - skipped")
            elif abs(new - old) > 1e-9:
                rate_changes[pid] = (old, new)

        for api_key, file_key in LIMIT_FIELDS.items():
            try:
                values = {to_number(i[api_key]) for i in items}
            except (KeyError, ValueError) as e:
                warnings.append(f"{pid}: unreadable {api_key} ({e}) - skipped")
                continue
            if len(values) > 1:
                warnings.append(f"{pid}: bank entries disagree on {file_key} {sorted(values)} - skipped")
                continue
            new, old = values.pop(), prog[file_key]
            if new <= 0:
                warnings.append(f"{pid}: implausible {file_key} {new} - skipped")
            elif new != float(old):
                limit_changes.setdefault(pid, {})[file_key] = (old, int(new) if new.is_integer() else new)
    return rate_changes, limit_changes, warnings

def _set_program_number(text: str, pid: str, field: str, value) -> str:
    """بيغير رقم حقل واحد جوه برنامج واحد في نص الملف، من غير ما يعيد كتابة الـ JSON كله."""
    pattern = re.compile(
        r'("id"\s*:\s*"' + re.escape(pid) + r'"(?:(?!"id"\s*:).)*?"'
        + re.escape(field) + r'"\s*:\s*)(-?\d+(?:\.\d+)?)',
        re.DOTALL,
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + repr(value), text)
    if n != 1:
        raise RuntimeError(f"{pid}.{field}: expected exactly 1 place to edit, found {n}")
    return new_text


def build_updated_text(text: str, rate_changes: dict) -> str:
    """
    بيغير رقم الفايدة بس جوه نص الملف، فباقي الملف والتنسيق بيفضلوا زي ما هم،
    وفرق git بيبقى سطر الفايدة بس. وبعدها بيتأكد إن الفرق الوحيد بين القديم
    والجديد هو الفايدة المطلوبة، ولو لقى غير كده بيلغي التعديل.
    """
    new_text = text
    for pid, (_, new_rate) in rate_changes.items():
        new_text = _set_program_number(new_text, pid, RATE_FIELD, float(new_rate))

    expected = json.loads(text)
    for prog in expected["programs"]:
        if prog["id"] in rate_changes:
            prog[RATE_FIELD] = float(rate_changes[prog["id"]][1])
    if json.loads(new_text) != expected:
        raise RuntimeError("verification failed: edit changed something other than the rate")
    return new_text



# START (1/2) ---------------------------------------------------------------
def build_limits_text(text: str, limit_changes: dict) -> str:
    """زي build_updated_text بس لحدود المبلغ والمدة (min/max loan و min/max tenor)."""
    new_text = text
    for pid, fields in limit_changes.items():
        for key, (_, new_value) in fields.items():
            new_text = _set_program_number(new_text, pid, key, new_value)
    expected = json.loads(text)
    for prog in expected["programs"]:
        for key, (_, new_value) in limit_changes.get(prog["id"], {}).items():
            prog[key] = new_value
    if json.loads(new_text) != expected:
        raise RuntimeError("verification failed: edit changed something other than the limits")
    return new_text
# END (1/2) -----------------------------------------------------------------


def backup(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync loan interest rates from Banque Misr API")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not edit files")
    args = parser.parse_args()
    exit_code = 0

    for name, cfg in LOANS.items():
        if cfg["type_id"] == "PUT_ID_HERE":
            log.error("%s: loan type id is not set (env var or LOANS config)", name)
            exit_code = 1
            continue
        try:
            api_items = fetch_programs(cfg["type_id"])
        except Exception as e:
            log.error("%s: fetch failed, rules file left untouched: %s", name, e)
            exit_code = 1
            continue

        path = RULES_DIR / cfg["rules_file"]
        text = path.read_bytes().decode("utf-8")
        file_programs = {p["id"]: p for p in json.loads(text)["programs"]}

        rate_changes, limit_changes, warnings = analyze(
            api_items, build_mapping(file_programs, cfg["extra_mapping"]), file_programs
        )
        for w in warnings:
            log.warning("%s: %s", name, w)
        for pid, (old, new) in rate_changes.items():
            log.warning("%s: %s [%s] rate %s -> %s", name, pid, file_programs[pid]["name"], old, new)
        for pid, fields in limit_changes.items():
            for key, (old, new) in fields.items():
                log.warning("%s: %s [%s] LIMIT %s: file=%s bank=%s", name, pid, file_programs[pid]["name"], key, old, new)

        try:
            new_text = text
            if rate_changes:
                new_text = build_updated_text(new_text, rate_changes)
            # START (2/2) -------------------------------------------------------
            if limit_changes:
                new_text = build_limits_text(new_text, limit_changes)
            # END (2/2) ---------------------------------------------------------
            if new_text == text:
                log.info("%s: no changes applied (%d bank programs checked)", name, len(api_items))
                continue
            if args.dry_run:
                log.info("%s: dry run, file not modified", name)
                continue
            saved = backup(path)
            write_atomic(path, new_text)
            log.info("%s: rules file updated. backup: %s", name, saved)
        except Exception as e:
            log.error("%s: update aborted, file untouched: %s", name, e)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
