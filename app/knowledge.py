from dataclasses import dataclass


NCI_TREATMENT_BY_STAGE = "https://www.cancer.gov/types/breast/treatment/by-stage"


@dataclass(frozen=True)
class Guideline:
    stage: str
    summary: str
    source_url: str = NCI_TREATMENT_BY_STAGE


# Educational paraphrases of the NCI patient summary, simplified for this demo.
# Real treatment depends on subtype, biomarkers, health, preferences, and many
# other factors; this mapping must never be used for care decisions.
GUIDELINES = {
    "I": Guideline(
        "I",
        "Early-stage disease usually begins with surgery. Radiation and systemic "
        "therapy may follow depending on the operation, biomarkers, and risk factors.",
    ),
    "II": Guideline(
        "II",
        "Treatment commonly combines surgery with radiation and/or systemic therapy. "
        "Larger tumors may receive chemotherapy or targeted therapy before surgery.",
    ),
    "III": Guideline(
        "III",
        "Locally advanced disease often begins with systemic therapy, followed by "
        "surgery and radiation when appropriate.",
    ),
    "IV": Guideline(
        "IV",
        "Metastatic treatment focuses on slowing disease and controlling symptoms. "
        "Options depend strongly on biomarkers, prior response, goals, and overall health.",
    ),
}


PASSAGES = [
    {
        "stage": "I",
        "passage": "Stages I and some stage II breast cancers are early-stage disease. Treatment usually starts with surgery; additional therapy may follow.",
        "source_url": NCI_TREATMENT_BY_STAGE,
    },
    {
        "stage": "II",
        "passage": "For a large early-stage tumor, chemotherapy or targeted therapy may be given before surgery to make removal easier.",
        "source_url": NCI_TREATMENT_BY_STAGE,
    },
    {
        "stage": "III",
        "passage": "Locally advanced breast cancer often begins with chemotherapy and is followed by surgery and radiation.",
        "source_url": NCI_TREATMENT_BY_STAGE,
    },
    {
        "stage": "IV",
        "passage": "Metastatic breast cancer treatment focuses on slowing spread and controlling symptoms; treatment choice depends on earlier response and patient goals.",
        "source_url": NCI_TREATMENT_BY_STAGE,
    },
]


def lookup_guideline(stage: str) -> Guideline:
    try:
        return GUIDELINES[stage]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage: {stage}") from exc

