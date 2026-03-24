"""Generate synthetic clinical notes for testing and demo purposes.

Produces realistic but entirely fictional clinical notes across multiple
specialties. No real patient data is used - this demonstrates HIPAA-aware
data handling by using only synthetic data in public-facing demos.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SPECIALTIES = [
    "Internal Medicine",
    "Cardiology",
    "Endocrinology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Orthopedics",
    "Dentistry",
    "Emergency Medicine",
    "Family Medicine",
]

CHIEF_COMPLAINTS = {
    "Internal Medicine": [
        "Follow-up for chronic conditions",
        "Fatigue and malaise",
        "Routine health maintenance",
    ],
    "Cardiology": [
        "Chest pain on exertion",
        "Palpitations and dizziness",
        "Follow-up for atrial fibrillation",
    ],
    "Endocrinology": [
        "Follow-up for diabetes mellitus type 2",
        "Thyroid nodule evaluation",
        "Uncontrolled blood sugars",
    ],
    "Pulmonology": [
        "Chronic cough for 6 weeks",
        "Shortness of breath on exertion",
        "COPD exacerbation",
    ],
    "Gastroenterology": [
        "Abdominal pain and bloating",
        "Rectal bleeding",
        "Chronic diarrhea",
    ],
    "Neurology": [
        "Recurring headaches",
        "Numbness in extremities",
        "Memory concerns",
    ],
    "Orthopedics": [
        "Low back pain for 3 months",
        "Knee pain after fall",
        "Shoulder impingement",
    ],
    "Dentistry": [
        "Tooth pain lower right",
        "Routine dental examination",
        "Bleeding gums and sensitivity",
    ],
    "Emergency Medicine": [
        "Fall with hip pain",
        "Acute abdominal pain",
        "Laceration to hand",
    ],
    "Family Medicine": [
        "Annual wellness visit",
        "Upper respiratory symptoms",
        "Medication refill and follow-up",
    ],
}

MEDICATIONS = [
    ("Metformin", "1000mg", "PO BID"),
    ("Lisinopril", "10mg", "PO daily"),
    ("Atorvastatin", "20mg", "PO QHS"),
    ("Aspirin", "81mg", "PO daily"),
    ("Omeprazole", "20mg", "PO daily"),
    ("Amlodipine", "5mg", "PO daily"),
    ("Metoprolol", "25mg", "PO BID"),
    ("Levothyroxine", "50mcg", "PO daily"),
    ("Prednisone", "10mg", "PO daily"),
    ("Amoxicillin", "500mg", "PO TID"),
    ("Ibuprofen", "400mg", "PO TID PRN"),
    ("Acetaminophen", "500mg", "PO Q6H PRN"),
]

DIAGNOSES = [
    ("E11.9", "Type 2 diabetes mellitus"),
    ("I10", "Essential hypertension"),
    ("E78.5", "Dyslipidemia"),
    ("J45.9", "Asthma"),
    ("M54.5", "Low back pain"),
    ("K21.0", "GERD"),
    ("F32.9", "Major depressive disorder"),
    ("G43.9", "Migraine"),
    ("I48.91", "Atrial fibrillation"),
    ("N18.9", "Chronic kidney disease"),
]


def generate_note(specialty: str | None = None) -> dict:
    """Generate a single synthetic clinical note."""
    if specialty is None:
        specialty = random.choice(SPECIALTIES)

    age = random.randint(25, 85)
    sex = random.choice(["male", "female"])
    cc = random.choice(CHIEF_COMPLAINTS.get(specialty, ["Follow-up"]))

    num_meds = random.randint(1, 5)
    meds = random.sample(MEDICATIONS, min(num_meds, len(MEDICATIONS)))

    num_dx = random.randint(1, 4)
    dx = random.sample(DIAGNOSES, min(num_dx, len(DIAGNOSES)))

    note = f"""CHIEF COMPLAINT: {cc}

HISTORY OF PRESENT ILLNESS:
Patient is a {age}-year-old {sex} presenting for {cc.lower()}.
{"Reports compliance with current medications." if random.random() > 0.3 else "Reports occasional non-compliance with medications."}
{"Denies chest pain, shortness of breath, or fever." if random.random() > 0.5 else "Reports mild fatigue and occasional headaches."}

PAST MEDICAL HISTORY:
{chr(10).join(f"- {d[1]}" for d in dx)}

MEDICATIONS:
{chr(10).join(f"- {m[0]} {m[1]} {m[2]}" for m in meds)}

ALLERGIES: {"NKDA" if random.random() > 0.3 else "Penicillin (rash)"}

PHYSICAL EXAMINATION:
Vitals: BP {random.randint(110, 160)}/{random.randint(65, 95)}, HR {random.randint(60, 100)}, T {random.uniform(97.5, 99.5):.1f}F, RR {random.randint(12, 20)}, SpO2 {random.randint(95, 100)}%
General: {"Well-appearing, no acute distress" if random.random() > 0.3 else "Appears mildly uncomfortable"}
HEENT: {"Normocephalic, atraumatic. Pupils equal and reactive." if specialty != "Dentistry" else "Oral exam: " + random.choice(["caries noted #19", "gingival inflammation noted", "WNL"])}

ASSESSMENT AND PLAN:
{chr(10).join(f"{i+1}. {d[1]} ({d[0]}) - {random.choice(['continue current management', 'adjust medications', 'order follow-up labs', 'refer to specialist'])}" for i, d in enumerate(dx))}

Follow up in {random.choice(["2 weeks", "1 month", "3 months", "6 months"])}.
"""

    date = datetime.now() - timedelta(days=random.randint(0, 365))

    return {
        "text": note.strip(),
        "specialty": specialty,
        "metadata": {
            "patient_age": age,
            "patient_sex": sex,
            "date": date.strftime("%Y-%m-%d"),
            "synthetic": True,
        },
    }


def generate_dataset(n: int = 100, output_path: str | None = None) -> list[dict]:
    """Generate a dataset of synthetic clinical notes."""
    notes = [generate_note() for _ in range(n)]

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(notes, f, indent=2)

    return notes


if __name__ == "__main__":
    output = "ml/data/synthetic/clinical_notes.json"
    notes = generate_dataset(200, output)
    print(f"Generated {len(notes)} synthetic clinical notes -> {output}")

    # Print specialty distribution
    from collections import Counter
    dist = Counter(n["specialty"] for n in notes)
    for spec, count in dist.most_common():
        print(f"  {spec}: {count}")
