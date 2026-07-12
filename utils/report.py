"""
Clinical text output based on predicted DR grade.
Maps grade 0-4 -> headline + clinical body text.

This module is the single source of truth for risk level and follow-up
timing. app.py should read report["risk"] / report["followup"] rather
than keeping its own separate copy, so the metric cards and the clinical
text block can never disagree with each other.
"""

CLINICAL_DATA = {
    0: {
        "headline": "No diabetic retinopathy detected",
        "body": (
            "No signs of diabetic retinopathy were found in this fundus image. "
            "Continue routine annual screening. Maintain good glycaemic control "
            "(HbA1c < 7%) and blood pressure management to prevent future onset."
        ),
        "risk": "Low",
        "followup": "12 months",
        "referral": False,
    },
    1: {
        "headline": "Mild NPDR — routine monitoring recommended",
        "body": (
            "Mild non-proliferative diabetic retinopathy has been detected, "
            "characterised by a few microaneurysms. No vision-threatening changes "
            "are present at this stage. Schedule a follow-up with an ophthalmologist "
            "within 6–12 months. Optimise blood glucose and blood pressure control."
        ),
        "risk": "Low-Moderate",
        "followup": "6–12 months",
        "referral": False,
    },
    2: {
        "headline": "Moderate NPDR — ophthalmologist referral recommended",
        "body": (
            "Moderate non-proliferative diabetic retinopathy has been detected. "
            "The highlighted regions in the Grad-CAM heatmap indicate areas of "
            "microaneurysms and possible haemorrhage. A follow-up appointment with "
            "an ophthalmologist is recommended within 3–6 months. "
            "Ensure HbA1c levels are reviewed and blood sugar is well-controlled."
        ),
        "risk": "Moderate",
        "followup": "3–6 months",
        "referral": True,
    },
    3: {
        "headline": "Severe NPDR — urgent ophthalmologist referral required",
        "body": (
            "Severe non-proliferative diabetic retinopathy has been detected. "
            "This stage is characterised by extensive microaneurysms, intraretinal "
            "haemorrhages, venous beading, or IRMA. There is a high risk of "
            "progression to proliferative DR. Urgent referral to an ophthalmologist "
            "within 1 month is strongly recommended. Pan-retinal photocoagulation "
            "may be considered."
        ),
        "risk": "High",
        "followup": "Within 1 month",
        "referral": True,
    },
    4: {
        "headline": "Proliferative DR — immediate specialist referral required",
        "body": (
            "Proliferative diabetic retinopathy has been detected. New blood vessel "
            "growth (neovascularisation) on the retina or optic disc represents a "
            "sight-threatening emergency. Immediate referral to a retinal specialist "
            "is required. Treatment options include pan-retinal photocoagulation, "
            "anti-VEGF injections, or vitrectomy. Do not delay."
        ),
        "risk": "Very High",
        "followup": "Immediate / Urgent",
        "referral": True,
    },
}


def get_clinical_report(grade: int, probs) -> dict:
    """
    Return clinical report dict for a given grade.

    Raises ValueError on an out-of-range grade instead of silently
    falling back to the "No DR" entry, since a silent fallback in a
    screening tool could tell a patient everything is fine when the
    real cause was a bug upstream (e.g. an unexpected class index).
    """
    if grade not in CLINICAL_DATA:
        raise ValueError(
            f"Unknown DR grade: {grade!r}. Expected one of {sorted(CLINICAL_DATA)}."
        )

    if probs is None or len(probs) <= grade:
        raise ValueError(
            f"probs has length {len(probs) if probs is not None else 'None'}, "
            f"but grade {grade} requires at least {grade + 1} entries."
        )

    report = CLINICAL_DATA[grade].copy()
    report["confidence"] = float(probs[grade]) * 100
    report["grade"] = grade
    return report