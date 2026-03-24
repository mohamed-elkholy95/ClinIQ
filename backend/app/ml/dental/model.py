"""Dental-specific NLP module.

Provides pattern-based entity extraction for dental clinical text (tooth
numbering, surfaces, procedures, periodontal measurements, findings) and a
rule-based periodontal risk assessor.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import InferenceError, ModelLoadError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dental entity type registry
# ---------------------------------------------------------------------------

DENTAL_ENTITY_TYPES: dict[str, str] = {
    # Tooth identification
    "TOOTH_NUMBER_UNIVERSAL": "Universal Numbering System (1–32)",
    "TOOTH_NUMBER_FDI": "FDI World Dental Federation notation (11–48)",
    "TOOTH_SURFACE": "Tooth surface (M, D, B, L, F, I, O)",
    # Procedures
    "DENTAL_PROCEDURE": "Dental procedure (extraction, restoration, etc.)",
    "DENTAL_MATERIAL": "Restorative material (composite, amalgam, crown, etc.)",
    # Periodontal measurements
    "PROBING_DEPTH": "Periodontal probing depth in mm",
    "ATTACHMENT_LOSS": "Clinical attachment loss (CAL) in mm",
    "BONE_LOSS": "Radiographic bone loss percentage or mm",
    "BLEEDING_ON_PROBING": "Bleeding on probing (BOP) event",
    "FURCATION": "Furcation involvement class (I, II, III)",
    # Findings
    "CARIES": "Carious lesion finding",
    "FRACTURE": "Tooth or restoration fracture",
    "ABSCESS": "Periapical or periodontal abscess",
    "CALCULUS": "Supra- or subgingival calculus",
    "RECESSION": "Gingival recession in mm",
}

# ---------------------------------------------------------------------------
# Tooth numbering reference tables
# ---------------------------------------------------------------------------

# Universal Numbering System: 1–32 (permanent dentition)
# 1=UR3M, 2=UR2M, 3=UR1M, 4=URPm2, 5=URPm1, 6=URC, 7=URI2, 8=URI1
# 9=ULI1, 10=ULI2, 11=ULC, 12=ULPm1, 13=ULPm2, 14=UL1M, 15=UL2M, 16=UL3M
# (lower arch mirror)
TOOTH_NUMBERING: dict[str, Any] = {
    "universal": {str(i): f"Tooth #{i} (Universal)" for i in range(1, 33)},
    # FDI two-digit notation: first digit = quadrant (1–4), second = position (1–8)
    "fdi": {
        f"{quadrant}{position}": f"FDI {quadrant}{position}"
        for quadrant in range(1, 5)
        for position in range(1, 9)
    },
    # Mapping from FDI to Universal (permanent teeth)
    "fdi_to_universal": {
        "18": "1",  "17": "2",  "16": "3",  "15": "4",  "14": "5",
        "13": "6",  "12": "7",  "11": "8",  "21": "9",  "22": "10",
        "23": "11", "24": "12", "25": "13", "26": "14", "27": "15",
        "28": "16", "38": "17", "37": "18", "36": "19", "35": "20",
        "34": "21", "33": "22", "32": "23", "31": "24", "41": "25",
        "42": "26", "43": "27", "44": "28", "45": "29", "46": "30",
        "47": "31", "48": "32",
    },
}

# Tooth surfaces (acronyms used in dental charting)
TOOTH_SURFACES: dict[str, str] = {
    "M": "Mesial",
    "D": "Distal",
    "B": "Buccal",
    "F": "Facial",
    "L": "Lingual",
    "P": "Palatal",
    "I": "Incisal",
    "O": "Occlusal",
    "MOD": "Mesio-occluso-distal",
    "MO": "Mesio-occlusal",
    "DO": "Disto-occlusal",
    "MOB": "Mesio-occluso-buccal",
}

# ---------------------------------------------------------------------------
# CDT code reference
# ---------------------------------------------------------------------------

CDT_CODES: dict[str, str] = {
    # Diagnostic
    "D0120": "Periodic oral evaluation",
    "D0140": "Limited oral evaluation – problem focused",
    "D0150": "Comprehensive oral evaluation",
    "D0180": "Comprehensive periodontal evaluation",
    "D0210": "Intraoral – complete series of radiographic images",
    "D0220": "Intraoral – periapical radiographic image",
    "D0230": "Intraoral – additional periapical radiographic image",
    "D0270": "Bitewing radiographic image",
    "D0330": "Panoramic radiographic image",
    # Preventive
    "D1110": "Prophylaxis – adult",
    "D1120": "Prophylaxis – child",
    "D1206": "Topical application of fluoride varnish",
    "D1351": "Sealant – per tooth",
    # Restorative
    "D2140": "Amalgam restoration – one surface, primary or permanent",
    "D2150": "Amalgam restoration – two surfaces",
    "D2160": "Amalgam restoration – three surfaces",
    "D2330": "Resin-based composite – one surface, anterior",
    "D2331": "Resin-based composite – two surfaces, anterior",
    "D2332": "Resin-based composite – three surfaces, anterior",
    "D2390": "Resin-based composite crown, anterior",
    "D2740": "Crown – porcelain/ceramic substrate",
    "D2750": "Crown – porcelain fused to high noble metal",
    "D2930": "Prefabricated stainless steel crown – primary tooth",
    "D2940": "Protective restoration",
    # Endodontic
    "D3110": "Pulp cap – direct",
    "D3120": "Pulp cap – indirect",
    "D3220": "Therapeutic pulpotomy",
    "D3310": "Endodontic therapy – anterior tooth",
    "D3320": "Endodontic therapy – premolar tooth",
    "D3330": "Endodontic therapy – molar tooth",
    # Periodontic
    "D4341": "Periodontal scaling and root planing – four or more teeth per quadrant",
    "D4342": "Periodontal scaling and root planing – one to three teeth per quadrant",
    "D4346": "Scaling in presence of generalized moderate or severe gingival inflammation",
    "D4355": "Full mouth debridement",
    "D4381": "Localized delivery of antimicrobial agents via a controlled release vehicle",
    "D4910": "Periodontal maintenance",
    # Prosthodontic
    "D5110": "Complete denture – maxillary",
    "D5120": "Complete denture – mandibular",
    "D5213": "Partial denture – maxillary",
    "D5214": "Partial denture – mandibular",
    "D6010": "Implant supported prosthetics",
    "D6065": "Implant supported porcelain/ceramic crown",
    # Oral surgery
    "D7140": "Extraction – erupted tooth or exposed root",
    "D7210": "Extraction – erupted tooth requiring elevation of mucoperiosteal flap",
    "D7240": "Removal of impacted tooth – completely bony",
    "D7250": "Removal of residual tooth roots",
    "D7310": "Alveoloplasty in conjunction with extractions",
    "D7410": "Excision of benign lesion up to 1.25 cm",
    "D7510": "Incision and drainage of abscess – intraoral soft tissue",
    "D7999": "Unspecified oral surgery procedure",
}

# ---------------------------------------------------------------------------
# Pattern helpers (compiled at module import time)
# ---------------------------------------------------------------------------

# Universal tooth number:  #1 – #32  or  tooth 1–32 (with optional leading zeros)
_UNIV_TOOTH_RE = re.compile(
    r"(?:#|tooth\s+)(?P<num>0?[1-9]|[12]\d|3[012])\b", re.IGNORECASE
)

# FDI notation:  two-digit codes 11–18, 21–28, 31–38, 41–48
_FDI_TOOTH_RE = re.compile(
    r"\b(?P<fdi>(?:[1-4][1-8]))\b"
)

# Surfaces – one or more surface codes, optionally joined
_SURFACE_RE = re.compile(
    r"\b(?P<surfaces>[MDBFLPIO]{1,5})\b"
    r"(?=\s*(?:surface|caries|restoration|cavity|fracture|margin))?",
    re.IGNORECASE,
)

# Probing depths:  "PD: 3 2 4" or "3mm" or "probing 4"
_PROBING_RE = re.compile(
    r"(?:pd|probing\s+depth|pocket\s+depth|probe)\s*:?\s*"
    r"(?P<depths>(?:\d+\s*)+)",
    re.IGNORECASE,
)
_PROBING_MM_RE = re.compile(
    r"\b(?P<depth>\d+)\s*mm\s*(?:probing|pocket|depth|pd)\b",
    re.IGNORECASE,
)

# Attachment loss / recession
_CAL_RE = re.compile(
    r"(?:cal|attachment\s+loss|clinical\s+attachment)\s*:?\s*(?P<mm>\d+)\s*mm",
    re.IGNORECASE,
)
_RECESSION_RE = re.compile(
    r"(?:recession|recede)\s*:?\s*(?P<mm>\d+)\s*mm",
    re.IGNORECASE,
)

# Bone loss
_BONE_LOSS_RE = re.compile(
    r"(?:bone\s+loss|alveolar\s+bone\s+loss)\s*:?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|mm)",
    re.IGNORECASE,
)

# Bleeding on probing
_BOP_RE = re.compile(
    r"\b(?:bop|bleeding\s+on\s+probing|bleed(?:ing)?)\b",
    re.IGNORECASE,
)

# Furcation involvement
_FURCATION_RE = re.compile(
    r"(?:furcation|furation)\s*:?\s*(?:class\s*|grade\s*)?(?P<class>[123III]+)",
    re.IGNORECASE,
)

# Findings
_CARIES_RE = re.compile(
    r"\b(?:caries|carious|decay|cavity|cavitation|demineraliz)\w*\b",
    re.IGNORECASE,
)
_FRACTURE_RE = re.compile(
    r"\b(?:fracture|crack|craze|split|chip)\w*\b",
    re.IGNORECASE,
)
_ABSCESS_RE = re.compile(
    r"\b(?:abscess|periapical\s+lesion|apical\s+abscess|parulis|sinus\s+tract|draining)\w*\b",
    re.IGNORECASE,
)
_CALCULUS_RE = re.compile(
    r"\b(?:calculus|tartar|deposits)\w*\b",
    re.IGNORECASE,
)

# Common dental procedures
_PROCEDURE_RE = re.compile(
    r"\b(?:"
    r"extraction|exodontia|pull"
    r"|restoration|filling|amalgam|composite|resin"
    r"|crown|cap"
    r"|root\s+canal|endodontic\s+therapy|pulpectomy|pulpotomy"
    r"|scaling|root\s+planing|debridement|prophylaxis|prophy"
    r"|implant"
    r"|denture|partial|bridge"
    r"|biopsy"
    r"|sealant"
    r"|bleach|whitening"
    r")\b",
    re.IGNORECASE,
)

# Dental materials
_MATERIAL_RE = re.compile(
    r"\b(?:"
    r"amalgam|composite|resin|ceramic|porcelain|zirconia|lithium\s+disilicate"
    r"|glass\s+ionomer|gic|compomer|flowable"
    r"|gold|cast\s+metal"
    r"|temporary|temp|cavit|ipc|fuji"
    r")\b",
    re.IGNORECASE,
)

# CDT code pattern
_CDT_RE = re.compile(r"\bD(?P<code>\d{4})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DentalEntity:
    """A single extracted dental entity.

    Attributes
    ----------
    tooth_number:
        Tooth identified (Universal or FDI notation), if applicable.
    surface:
        Surface code(s) (e.g. ``"MOD"``), if applicable.
    procedure:
        Dental procedure name, if applicable.
    finding:
        Clinical finding (caries, fracture, abscess, …), if applicable.
    confidence:
        Extraction confidence in ``[0, 1]``.
    entity_type:
        Key from :data:`DENTAL_ENTITY_TYPES`.
    text_span:
        The raw matched text span.
    start_char:
        Start character offset in the original text.
    end_char:
        End character offset in the original text.
    metadata:
        Free-form metadata dict.
    """

    tooth_number: str | None
    surface: str | None
    procedure: str | None
    finding: str | None
    confidence: float
    entity_type: str = ""
    text_span: str = ""
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "tooth_number": self.tooth_number,
            "surface": self.surface,
            "procedure": self.procedure,
            "finding": self.finding,
            "confidence": self.confidence,
            "entity_type": self.entity_type,
            "text_span": self.text_span,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


@dataclass
class DentalAssessment:
    """Complete dental assessment result for a clinical note.

    Attributes
    ----------
    entities:
        All extracted :class:`DentalEntity` objects.
    periodontal_risk_score:
        Composite periodontal risk score in ``[0, 100]``.
    periodontal_classification:
        Classification string (e.g. ``"Stage II Grade B Periodontitis"``).
    cdt_codes:
        Suggested CDT codes with descriptions.
    recommendations:
        Clinical recommendations derived from the assessment.
    processing_time_ms:
        Wall-clock inference time in milliseconds.
    """

    entities: list[DentalEntity]
    periodontal_risk_score: float
    periodontal_classification: str
    cdt_codes: dict[str, str]
    recommendations: list[str]
    processing_time_ms: float
    model_name: str = ""
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "periodontal_risk_score": self.periodontal_risk_score,
            "periodontal_classification": self.periodontal_classification,
            "cdt_codes": self.cdt_codes,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# DentalNERModel
# ---------------------------------------------------------------------------


class DentalNERModel:
    """Pattern-based dental entity extraction.

    Extracts the following entity types from clinical dental notes:
    * Universal tooth numbers (``#1``–``#32``)
    * FDI tooth numbers (``11``–``48``)
    * Tooth surfaces (MDBLFIO and combinations)
    * Probing depths
    * Attachment loss / recession measurements
    * Bone loss
    * Bleeding-on-probing events
    * Furcation involvement
    * Dental findings (caries, fracture, abscess, calculus)
    * Dental procedures
    * Dental materials
    * CDT codes (D0120–D7999)
    """

    def __init__(
        self,
        model_name: str = "dental-pattern-ner",
        version: str = "1.0.0",
    ) -> None:
        self.model_name = model_name
        self.version = version
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise the model (patterns are pre-compiled at import time)."""
        self._is_loaded = True
        logger.info("Loaded DentalNERModel v%s", self.version)

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model has been loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Load if not yet loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[DentalEntity]:
        """Extract all dental entities from *text*.

        Returns
        -------
        list[DentalEntity]
            Entities in approximate document order.
        """
        self.ensure_loaded()

        try:
            entities: list[DentalEntity] = []

            entities.extend(self._extract_tooth_numbers(text))
            entities.extend(self._extract_surfaces(text))
            entities.extend(self._extract_probing_depths(text))
            entities.extend(self._extract_attachment_measurements(text))
            entities.extend(self._extract_bone_loss(text))
            entities.extend(self._extract_bop(text))
            entities.extend(self._extract_furcation(text))
            entities.extend(self._extract_findings(text))
            entities.extend(self._extract_procedures(text))
            entities.extend(self._extract_materials(text))
            entities.extend(self._extract_cdt_codes(text))

            # Sort by document position
            entities.sort(key=lambda e: e.start_char)
            return entities

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_tooth_numbers(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _UNIV_TOOTH_RE.finditer(text):
            num = match.group("num")
            entities.append(
                DentalEntity(
                    tooth_number=num,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.95,
                    entity_type="TOOTH_NUMBER_UNIVERSAL",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        for match in _FDI_TOOTH_RE.finditer(text):
            fdi = match.group("fdi")
            # Only accept valid FDI codes (quadrant 1–4, position 1–8)
            if len(fdi) == 2 and fdi[0] in "1234" and fdi[1] in "12345678":
                entities.append(
                    DentalEntity(
                        tooth_number=fdi,
                        surface=None,
                        procedure=None,
                        finding=None,
                        confidence=0.85,
                        entity_type="TOOTH_NUMBER_FDI",
                        text_span=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                        metadata={
                            "universal": TOOTH_NUMBERING["fdi_to_universal"].get(fdi)
                        },
                    )
                )

        return entities

    def _extract_surfaces(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []
        valid_surfaces = set(TOOTH_SURFACES.keys())

        for match in _SURFACE_RE.finditer(text):
            raw = match.group("surfaces").upper()
            # Only emit if the code is a known surface or a known combination
            if raw in valid_surfaces or all(ch in "MDBFLPIOA" for ch in raw):
                entities.append(
                    DentalEntity(
                        tooth_number=None,
                        surface=raw,
                        procedure=None,
                        finding=None,
                        confidence=0.80,
                        entity_type="TOOTH_SURFACE",
                        text_span=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                        metadata={"surface_name": TOOTH_SURFACES.get(raw, raw)},
                    )
                )

        return entities

    def _extract_probing_depths(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _PROBING_RE.finditer(text):
            depths_str = match.group("depths").strip()
            depths = [int(d) for d in depths_str.split() if d.isdigit()]
            if depths:
                entities.append(
                    DentalEntity(
                        tooth_number=None,
                        surface=None,
                        procedure=None,
                        finding=None,
                        confidence=0.9,
                        entity_type="PROBING_DEPTH",
                        text_span=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                        metadata={"depths_mm": depths, "max_depth_mm": max(depths)},
                    )
                )

        for match in _PROBING_MM_RE.finditer(text):
            depth = int(match.group("depth"))
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.85,
                    entity_type="PROBING_DEPTH",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"depths_mm": [depth], "max_depth_mm": depth},
                )
            )

        return entities

    def _extract_attachment_measurements(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _CAL_RE.finditer(text):
            mm = int(match.group("mm"))
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.9,
                    entity_type="ATTACHMENT_LOSS",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"cal_mm": mm},
                )
            )

        for match in _RECESSION_RE.finditer(text):
            mm = int(match.group("mm"))
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.9,
                    entity_type="RECESSION",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"recession_mm": mm},
                )
            )

        return entities

    def _extract_bone_loss(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _BONE_LOSS_RE.finditer(text):
            val = float(match.group("val"))
            unit = match.group("unit")
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.9,
                    entity_type="BONE_LOSS",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"value": val, "unit": unit},
                )
            )

        return entities

    def _extract_bop(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _BOP_RE.finditer(text):
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding="bleeding on probing",
                    confidence=0.85,
                    entity_type="BLEEDING_ON_PROBING",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        return entities

    def _extract_furcation(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _FURCATION_RE.finditer(text):
            furcation_class = match.group("class")
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=f"Furcation class {furcation_class}",
                    confidence=0.88,
                    entity_type="FURCATION",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"furcation_class": furcation_class},
                )
            )

        return entities

    def _extract_findings(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []
        finding_patterns: list[tuple[re.Pattern[str], str]] = [
            (_CARIES_RE, "CARIES"),
            (_FRACTURE_RE, "FRACTURE"),
            (_ABSCESS_RE, "ABSCESS"),
            (_CALCULUS_RE, "CALCULUS"),
        ]
        for pattern, etype in finding_patterns:
            for match in pattern.finditer(text):
                entities.append(
                    DentalEntity(
                        tooth_number=None,
                        surface=None,
                        procedure=None,
                        finding=match.group().lower(),
                        confidence=0.85,
                        entity_type=etype,
                        text_span=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

        return entities

    def _extract_procedures(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _PROCEDURE_RE.finditer(text):
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=match.group().lower(),
                    finding=None,
                    confidence=0.88,
                    entity_type="DENTAL_PROCEDURE",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        return entities

    def _extract_materials(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _MATERIAL_RE.finditer(text):
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=None,
                    finding=None,
                    confidence=0.85,
                    entity_type="DENTAL_MATERIAL",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        return entities

    def _extract_cdt_codes(self, text: str) -> list[DentalEntity]:
        entities: list[DentalEntity] = []

        for match in _CDT_RE.finditer(text):
            code = "D" + match.group("code")
            description = CDT_CODES.get(code, "Unknown CDT code")
            entities.append(
                DentalEntity(
                    tooth_number=None,
                    surface=None,
                    procedure=description,
                    finding=None,
                    confidence=0.99,
                    entity_type="DENTAL_PROCEDURE",
                    text_span=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"cdt_code": code, "description": description},
                )
            )

        return entities


# ---------------------------------------------------------------------------
# PeriodontalRiskAssessor
# ---------------------------------------------------------------------------

# 2017 AAP/EFP classification thresholds
# Stage I:  PD ≤4 mm, CAL 1–2 mm, bone loss <15%
# Stage II: PD ≤5 mm, CAL 3–4 mm, bone loss 15–33%
# Stage III: PD ≥6 mm, CAL ≥5 mm, bone loss ≥33%
# Stage IV:  Stage III + complexity factors

_PERIO_CLASSIFICATION_THRESHOLDS: list[tuple[str, float]] = [
    ("Healthy", 10.0),
    ("Gingivitis", 25.0),
    ("Stage I Periodontitis (Mild)", 40.0),
    ("Stage II Periodontitis (Moderate)", 60.0),
    ("Stage III Periodontitis (Severe)", 80.0),
    ("Stage IV Periodontitis (Very Severe)", 100.0),
]


class PeriodontalRiskAssessor:
    """Rule-based periodontal risk assessor.

    Scores based on:
    * **Probing depths** – depths ≥4 mm carry risk; ≥6 mm carry critical risk
    * **Bleeding on probing (BOP)** – presence increases inflammatory burden
    * **Clinical attachment loss (CAL)** – higher loss → higher stage
    * **Bone loss** – percentage or mm value
    * **Furcation involvement** – class II/III significantly elevates risk
    * **Calculus** – heavy deposits suggest poor control

    Classification follows the 2017 AAP/EFP World Workshop criteria
    (Papapanou et al., 2018).
    """

    def __init__(
        self,
        model_name: str = "periodontal-risk",
        version: str = "1.0.0",
    ) -> None:
        self.model_name = model_name
        self.version = version
        self._is_loaded: bool = False
        self._ner: DentalNERModel = DentalNERModel()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the NER model used for measurement extraction."""
        self._ner.load()
        self._is_loaded = True
        logger.info("Loaded PeriodontalRiskAssessor v%s", self.version)

    @property
    def is_loaded(self) -> bool:
        """``True`` if loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Load if not yet loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assess(self, text: str, entities: list[DentalEntity] | None = None) -> dict[str, Any]:
        """Assess periodontal risk and classify disease severity.

        Parameters
        ----------
        text:
            Raw dental clinical note.
        entities:
            Pre-extracted :class:`DentalEntity` list; if *None*, entities are
            extracted automatically from *text*.

        Returns
        -------
        dict
            Keys: ``risk_score`` (0–100), ``classification``,
            ``findings``, ``recommendations``.
        """
        self.ensure_loaded()
        start_time = time.time()

        try:
            if entities is None:
                entities = self._ner.extract_entities(text)

            score, findings = self._compute_perio_score(entities)
            classification = self._classify(score)
            recommendations = self._generate_recommendations(
                score, classification, findings
            )

            return {
                "risk_score": round(score, 2),
                "classification": classification,
                "findings": findings,
                "recommendations": recommendations,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_perio_score(
        self, entities: list[DentalEntity]
    ) -> tuple[float, dict[str, Any]]:
        """Return (score 0–100, findings dict)."""
        score = 0.0
        findings: dict[str, Any] = {}

        # --- Probing depths ---
        all_depths: list[int] = []
        for ent in entities:
            if ent.entity_type == "PROBING_DEPTH":
                all_depths.extend(ent.metadata.get("depths_mm", []))
                if ent.metadata.get("max_depth_mm") is not None:
                    all_depths.append(ent.metadata["max_depth_mm"])

        if all_depths:
            max_pd = max(all_depths)
            mean_pd = sum(all_depths) / len(all_depths)
            sites_ge4 = sum(1 for d in all_depths if d >= 4)
            sites_ge6 = sum(1 for d in all_depths if d >= 6)
            findings["max_probing_depth_mm"] = max_pd
            findings["mean_probing_depth_mm"] = round(mean_pd, 1)
            findings["sites_ge4mm"] = sites_ge4
            findings["sites_ge6mm"] = sites_ge6

            # Score contribution from PD (0–35 points)
            if max_pd >= 6:
                score += 35.0
            elif max_pd >= 5:
                score += 22.0
            elif max_pd >= 4:
                score += 12.0
            else:
                score += max(0.0, (mean_pd - 2.0) * 4.0)

        # --- Bleeding on probing ---
        bop_count = sum(
            1 for e in entities if e.entity_type == "BLEEDING_ON_PROBING"
        )
        findings["bop_sites"] = bop_count
        score += min(15.0, bop_count * 3.0)

        # --- Clinical attachment loss ---
        cal_values: list[int] = []
        for ent in entities:
            if ent.entity_type == "ATTACHMENT_LOSS":
                cal_values.append(ent.metadata.get("cal_mm", 0))

        if cal_values:
            max_cal = max(cal_values)
            findings["max_cal_mm"] = max_cal
            # Score contribution (0–25 points)
            if max_cal >= 5:
                score += 25.0
            elif max_cal >= 3:
                score += 15.0
            elif max_cal >= 1:
                score += 8.0

        # --- Bone loss ---
        bone_loss_vals: list[float] = []
        for ent in entities:
            if ent.entity_type == "BONE_LOSS":
                val = ent.metadata.get("value", 0)
                unit = ent.metadata.get("unit", "mm")
                if unit == "%":
                    bone_loss_vals.append(float(val))
                else:
                    # Convert mm to rough percentage (typical alveolar bone ~14 mm)
                    bone_loss_vals.append(float(val) / 14.0 * 100)

        if bone_loss_vals:
            max_bone_loss_pct = max(bone_loss_vals)
            findings["max_bone_loss_pct"] = round(max_bone_loss_pct, 1)
            # Score contribution (0–15 points)
            if max_bone_loss_pct >= 33:
                score += 15.0
            elif max_bone_loss_pct >= 15:
                score += 8.0
            else:
                score += 3.0

        # --- Furcation involvement ---
        furcation_classes: list[str] = []
        for ent in entities:
            if ent.entity_type == "FURCATION":
                furcation_classes.append(str(ent.metadata.get("furcation_class", "")))

        if furcation_classes:
            findings["furcation_involvement"] = furcation_classes
            if any(c in ("3", "III") for c in furcation_classes):
                score += 10.0
            elif any(c in ("2", "II") for c in furcation_classes):
                score += 6.0
            else:
                score += 2.0

        # --- Calculus / recession ---
        calculus_count = sum(1 for e in entities if e.entity_type == "CALCULUS")
        recession_count = sum(1 for e in entities if e.entity_type == "RECESSION")
        if calculus_count:
            score += min(5.0, calculus_count * 1.5)
        if recession_count:
            score += min(5.0, recession_count * 1.5)

        return min(100.0, score), findings

    def _classify(self, score: float) -> str:
        """Map score to a classification string."""
        for label, threshold in _PERIO_CLASSIFICATION_THRESHOLDS:
            if score <= threshold:
                return label
        return "Stage IV Periodontitis (Very Severe)"

    def _generate_recommendations(
        self,
        score: float,
        classification: str,
        findings: dict[str, Any],
    ) -> list[str]:
        """Produce clinical recommendations based on the periodontal assessment."""
        recommendations: list[str] = []

        if score >= 80:
            recommendations.append(
                "Immediate periodontal specialist referral indicated (Stage III/IV)"
            )
            recommendations.append(
                "Full-mouth debridement followed by periodontal surgery evaluation"
            )
        elif score >= 60:
            recommendations.append(
                "Comprehensive periodontal treatment required: scaling and root planing"
            )
            recommendations.append(
                "Re-evaluate in 6–8 weeks following initial therapy"
            )
        elif score >= 40:
            recommendations.append(
                "Periodontal scaling and root planing for affected quadrants (D4341/D4342)"
            )
            recommendations.append("3-month periodontal maintenance intervals recommended")
        elif score >= 25:
            recommendations.append(
                "Full-mouth prophylaxis and oral hygiene instruction (D1110)"
            )
            recommendations.append("6-month recall with re-evaluation of gingival status")
        else:
            recommendations.append("Continue routine preventive care (D0120/D1110)")

        # Specific finding-driven recommendations
        max_pd = findings.get("max_probing_depth_mm", 0)
        if max_pd >= 6:
            recommendations.append(
                f"Surgical consultation warranted: max probing depth {max_pd} mm"
            )

        bop = findings.get("bop_sites", 0)
        if bop >= 4:
            recommendations.append(
                "Intensive oral hygiene instruction; consider chlorhexidine rinse"
            )

        if "furcation_involvement" in findings:
            recommendations.append(
                "Furcation involvement documented — consider surgical or extraction planning"
            )

        return recommendations[:5]
