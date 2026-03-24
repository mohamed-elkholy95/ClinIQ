"""Dental-specific Named Entity Recognition module.

This module provides specialized NER for dental clinical notes,
including tooth numbering, surfaces, procedures, and periodontal
measurements. This is a unique differentiator leveraging clinical
dental expertise.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.ml.ner.model import Entity

logger = logging.getLogger(__name__)


class ToothNumberingSystem(str, Enum):
    """Supported tooth numbering systems."""

    UNIVERSAL = "universal"  # 1-32 (US system)
    FDI = "fdi"  # Two-digit system (international)
    PALMER = "palmer"  # Quadrant + number with symbols


class DentalSurface(str, Enum):
    """Tooth surfaces."""

    M = "mesial"
    D = "distal"
    B = "buccal"
    L = "lingual"
    O = "occlusal"
    I = "incisal"
    F = "facial"
    P = "palatal"
    MB = "mesiobuccal"
    ML = "mesiolingual"
    DB = "distobuccal"
    DL = "distolingual"


@dataclass
class DentalEntity(Entity):
    """Extended entity for dental-specific attributes."""

    tooth_number: str | None = None
    tooth_surface: list[str] | None = None
    numbering_system: str | None = None
    periodontal_value: float | None = None
    quadrant: int | None = None


# Dental procedure patterns
DENTAL_PROCEDURES = {
    # Restorative
    "amalgam": "restoration_amalgam",
    "composite": "restoration_composite",
    "filling": "restoration_filling",
    "crown": "prosthesis_crown",
    "bridge": "prosthesis_bridge",
    "veneer": "prosthesis_veneer",
    "onlay": "restoration_onlay",
    "inlay": "restoration_inlay",
    # Endodontic
    "root canal": "endodontic_rct",
    "rct": "endodontic_rct",
    "pulpectomy": "endodontic_pulpectomy",
    "pulpotomy": "endodontic_pulpotomy",
    # Periodontal
    "scaling": "periodontal_scaling",
    "root planing": "periodontal_root_planing",
    "srp": "periodontal_srp",
    "gingivectomy": "periodontal_gingivectomy",
    "gingivoplasty": "periodontal_gingivoplasty",
    "flap surgery": "periodontal_flap",
    "bone graft": "periodontal_bone_graft",
    # Surgical
    "extraction": "surgery_extraction",
    "exodontia": "surgery_extraction",
    "impaction": "surgery_impaction",
    "odontectomy": "surgery_odontectomy",
    "apicoectomy": "surgery_apicoectomy",
    "implant": "prosthesis_implant",
    # Orthodontic
    "braces": "ortho_fixed",
    "aligner": "ortho_aligner",
    "retainer": "ortho_retainer",
    "banding": "ortho_banding",
    "bracket": "ortho_bracket",
    # Preventive
    "prophylaxis": "preventive_prophy",
    "cleaning": "preventive_cleaning",
    "fluoride": "preventive_fluoride",
    "sealant": "preventive_sealant",
    # Diagnostic
    "bitewing": "diagnostic_bitewing",
    "periapical": "diagnostic_pa",
    "panoramic": "diagnostic_pano",
    "cbct": "diagnostic_cbct",
}

# Periodontal measurement patterns
PERIO_PATTERNS = {
    "pocket_depth": r"(\d+)\s*mm\s*(?:pocket|pd|depth)",
    "clinical_attachment": r"cal\s*[:=]?\s*(\d+)",
    "bleeding_on_probing": r"\b(?:bop|bleeding)\s*(?:\+|present|yes|\d+%?)",
    "plaque_index": r"\b(?:pi|plaque)\s*[:=]?\s*(\d)",
    "gingival_index": r"\b(?:gi|gingival)\s*[:=]?\s*(\d)",
    "mobility": r"mobility\s*[:=]?\s*(\d|grade\s*[i]+)",
    "furcation": r"furcation\s*[:=]?\s*(\d|grade\s*[i]+)",
}

# Tooth numbering patterns
TOOTH_PATTERNS = {
    "universal": r"\b(?:tooth\s*)?(?:#?\s*)([1-9]|1[0-9]|2[0-9]|3[0-2])\b",
    "fdi": r"\b([1-4][1-8])\b",  # 11-48
    "palmer": r"\b([a-j]|[1-8])([↙↘↖↗])\b",  # With quadrant symbols
    "letter": r"\b(?:tooth\s*)?([A-T])\b",  # Primary teeth
}


class DentalNERModel:
    """Specialized NER model for dental clinical notes."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self._compiled_patterns: dict[str, re.Pattern] = {}
        self._is_loaded = False

    def load(self) -> None:
        """Load and compile patterns."""
        # Compile tooth patterns
        for name, pattern in TOOTH_PATTERNS.items():
            self._compiled_patterns[f"tooth_{name}"] = re.compile(
                pattern, re.IGNORECASE
            )

        # Compile periodontal patterns
        for name, pattern in PERIO_PATTERNS.items():
            self._compiled_patterns[f"perio_{name}"] = re.compile(
                pattern, re.IGNORECASE
            )

        # Compile procedure patterns
        for proc, entity_type in DENTAL_PROCEDURES.items():
            pattern = r"\b" + re.escape(proc) + r"\b"
            self._compiled_patterns[f"proc_{proc}"] = re.compile(
                pattern, re.IGNORECASE
            )

        # Surface patterns
        surface_pattern = r"\b([MBLDIOF]{1,3})\b"
        self._compiled_patterns["surface"] = re.compile(surface_pattern)

        self._is_loaded = True
        logger.info(f"Dental NER model v{self.version} loaded")

    def extract_entities(self, text: str) -> list[DentalEntity]:
        """Extract dental-specific entities from text."""
        if not self._is_loaded:
            self.load()

        entities: list[DentalEntity] = []

        # Extract teeth
        entities.extend(self._extract_teeth(text))

        # Extract procedures
        entities.extend(self._extract_procedures(text))

        # Extract periodontal measurements
        entities.extend(self._extract_periodontal(text))

        # Extract surfaces
        entities.extend(self._extract_surfaces(text, entities))

        # Extract conditions
        entities.extend(self._extract_conditions(text))

        # Sort and resolve overlaps
        entities = self._resolve_overlaps(entities)

        # Detect negation
        entities = self._detect_negation(text, entities)

        return entities

    def _extract_teeth(self, text: str) -> list[DentalEntity]:
        """Extract tooth number references."""
        entities = []

        # Universal numbering (1-32)
        for match in self._compiled_patterns["tooth_universal"].finditer(text):
            tooth_num = match.group(1)
            quadrant = self._get_quadrant_universal(int(tooth_num))

            entities.append(
                DentalEntity(
                    text=f"#{tooth_num}",
                    entity_type="TOOTH",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.95,
                    tooth_number=tooth_num,
                    numbering_system="universal",
                    quadrant=quadrant,
                )
            )

        # FDI numbering (11-48)
        for match in self._compiled_patterns["tooth_fdi"].finditer(text):
            fdi_code = match.group(1)
            quadrant = int(fdi_code[0])

            entities.append(
                DentalEntity(
                    text=fdi_code,
                    entity_type="TOOTH",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.95,
                    tooth_number=fdi_code,
                    numbering_system="fdi",
                    quadrant=quadrant,
                )
            )

        # Primary teeth (A-T)
        for match in self._compiled_patterns["tooth_letter"].finditer(text):
            letter = match.group(1).upper()

            entities.append(
                DentalEntity(
                    text=letter,
                    entity_type="TOOTH_PRIMARY",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.90,
                    tooth_number=letter,
                    numbering_system="letter",
                )
            )

        return entities

    def _extract_procedures(self, text: str) -> list[DentalEntity]:
        """Extract dental procedure mentions."""
        entities = []

        for proc_name, entity_type in DENTAL_PROCEDURES.items():
            pattern_key = f"proc_{proc_name}"
            if pattern_key not in self._compiled_patterns:
                continue

            for match in self._compiled_patterns[pattern_key].finditer(text):
                entities.append(
                    DentalEntity(
                        text=match.group(),
                        entity_type="DENTAL_PROCEDURE",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.90,
                        metadata={"procedure_type": entity_type},
                    )
                )

        return entities

    def _extract_periodontal(self, text: str) -> list[DentalEntity]:
        """Extract periodontal measurements."""
        entities = []

        for meas_type, pattern_key in [(k, f"perio_{k}") for k in PERIO_PATTERNS]:
            if pattern_key not in self._compiled_patterns:
                continue

            for match in self._compiled_patterns[pattern_key].finditer(text):
                value = match.group(1) if match.groups() else None

                entities.append(
                    DentalEntity(
                        text=match.group(),
                        entity_type="PERIODONTAL",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.85,
                        metadata={
                            "measurement_type": meas_type,
                            "value": float(value) if value else None,
                        },
                        periodontal_value=float(value) if value else None,
                    )
                )

        return entities

    def _extract_surfaces(
        self, text: str, existing_entities: list[DentalEntity]
    ) -> list[DentalEntity]:
        """Extract tooth surface references."""
        entities = []

        for match in self._compiled_patterns["surface"].finditer(text):
            surface_code = match.group(1).upper()

            # Validate surface combinations
            if self._is_valid_surface(surface_code):
                entities.append(
                    DentalEntity(
                        text=surface_code,
                        entity_type="TOOTH_SURFACE",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.85,
                        tooth_surface=list(surface_code),
                    )
                )

        return entities

    def _extract_conditions(self, text: str) -> list[DentalEntity]:
        """Extract dental conditions and findings."""
        conditions = {
            "caries": "caries",
            "cavity": "caries",
            "decay": "caries",
            "caries": "caries",
            "fracture": "fracture",
            "crack": "fracture",
            "abscess": "abscess",
            "periapical lesion": "lesion",
            "cyst": "cyst",
            "gingivitis": "gingivitis",
            "periodontitis": "periodontitis",
            "recession": "gingival_recession",
            "attrition": "attrition",
            "abrasion": "abrasion",
            "erosion": "erosion",
            "hypersensitivity": "sensitivity",
            "impacted": "impaction",
            "malocclusion": "malocclusion",
            "diastema": "diastema",
            "supernumerary": "supernumerary",
        }

        entities = []
        for condition, entity_type in conditions.items():
            pattern = r"\b" + re.escape(condition) + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    DentalEntity(
                        text=match.group(),
                        entity_type="DENTAL_CONDITION",
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.85,
                        metadata={"condition": entity_type},
                    )
                )

        return entities

    def _get_quadrant_universal(self, tooth_num: int) -> int:
        """Get quadrant for universal tooth number."""
        if 1 <= tooth_num <= 8:
            return 1  # Upper right
        elif 9 <= tooth_num <= 16:
            return 2  # Upper left
        elif 17 <= tooth_num <= 24:
            return 3  # Lower left
        elif 25 <= tooth_num <= 32:
            return 4  # Lower right
        return 0

    def _is_valid_surface(self, code: str) -> bool:
        """Validate surface combination."""
        valid_surfaces = {"M", "D", "B", "L", "O", "I", "F", "P", "MB", "ML", "DB", "DL"}
        return code in valid_surfaces

    def _resolve_overlaps(self, entities: list[DentalEntity]) -> list[DentalEntity]:
        """Resolve overlapping entities, keeping highest confidence."""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda e: (e.start_char, -e.confidence))

        resolved = []
        for entity in entities:
            overlaps = False
            for existing in resolved:
                if self._overlaps(entity, existing):
                    overlaps = True
                    break
            if not overlaps:
                resolved.append(entity)

        return resolved

    def _overlaps(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap."""
        return e1.start_char < e2.end_char and e2.start_char < e1.end_char

    def _detect_negation(
        self, text: str, entities: list[DentalEntity]
    ) -> list[DentalEntity]:
        """Detect negated findings."""
        negation_pattern = r"\b(?:no|without|negative for|absence of)\s+"

        for entity in entities:
            # Check context before entity
            context_start = max(0, entity.start_char - 30)
            context = text[context_start : entity.start_char].lower()

            if re.search(negation_pattern, context):
                entity.is_negated = True

        return entities


class PeriodontalRiskAssessment:
    """Assess periodontal disease risk from clinical notes."""

    def __init__(self):
        self.version = "1.0.0"

    def calculate_risk(
        self,
        entities: list[DentalEntity],
        text: str,
    ) -> dict[str, Any]:
        """Calculate periodontal risk score."""
        risk_factors = []
        protective_factors = []

        # Extract periodontal measurements
        perio_entities = [e for e in entities if e.entity_type == "PERIODONTAL"]

        pocket_depths = []
        for e in perio_entities:
            if e.metadata and e.metadata.get("measurement_type") == "pocket_depth":
                if e.periodontal_value:
                    pocket_depths.append(e.periodontal_value)

        # Assess based on pocket depths
        if pocket_depths:
            avg_depth = sum(pocket_depths) / len(pocket_depths)
            max_depth = max(pocket_depths)

            if avg_depth > 4:
                risk_factors.append(
                    {
                        "factor": "elevated_average_pocket_depth",
                        "value": avg_depth,
                        "severity": "moderate" if avg_depth < 5 else "high",
                    }
                )

            if max_depth >= 6:
                risk_factors.append(
                    {
                        "factor": "deep_probing_depths",
                        "value": max_depth,
                        "severity": "high",
                    }
                )

        # Check for bleeding
        text_lower = text.lower()
        if "bleeding" in text_lower or "bop+" in text_lower:
            risk_factors.append(
                {"factor": "bleeding_on_probing", "severity": "moderate"}
            )

        # Check for mobility
        if "mobility" in text_lower and ("grade" in text_lower or any(str(i) in text_lower for i in range(1, 4))):
            risk_factors.append({"factor": "tooth_mobility", "severity": "high"})

        # Check for bone loss mentions
        if "bone loss" in text_lower:
            risk_factors.append({"factor": "radiographic_bone_loss", "severity": "high"})

        # Check for diagnosis
        if "periodontitis" in text_lower:
            risk_factors.append({"factor": "periodontitis_diagnosis", "severity": "high"})

        # Calculate overall risk
        if not risk_factors:
            overall_risk = "low"
            risk_score = 0.2
        elif len(risk_factors) == 1:
            overall_risk = "moderate"
            risk_score = 0.5
        else:
            high_severity = sum(1 for r in risk_factors if r["severity"] == "high")
            if high_severity >= 2:
                overall_risk = "high"
                risk_score = 0.8
            else:
                overall_risk = "moderate"
                risk_score = 0.6

        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommendations": self._generate_recommendations(risk_factors),
        }

    def _generate_recommendations(self, risk_factors: list[dict]) -> list[str]:
        """Generate clinical recommendations."""
        recommendations = []

        if any(r["factor"] == "deep_probing_depths" for r in risk_factors):
            recommendations.append("Consider referral to periodontist for comprehensive evaluation")

        if any(r["factor"] == "bleeding_on_probing" for r in risk_factors):
            recommendations.append("Enhance oral hygiene instructions and increase prophylaxis frequency")

        if any(r["factor"] == "periodontitis_diagnosis" for r in risk_factors):
            recommendations.append("Comprehensive periodontal evaluation and treatment planning indicated")

        if not recommendations:
            recommendations.append("Continue routine periodontal maintenance")

        return recommendations


class CDTCodePredictor:
    """Predict CDT dental procedure codes from clinical text."""

    # Simplified CDT code mapping
    CDT_CODES = {
        "cleaning": {"code": "D1110", "description": "Prophylaxis - adult"},
        "prophylaxis": {"code": "D1110", "description": "Prophylaxis - adult"},
        "scaling": {"code": "D4341", "description": "Periodontal scaling - quadrant"},
        "root planing": {"code": "D4342", "description": "Root planing - quadrant"},
        "amalgam": {"code": "D2391", "description": "Resin-based composite - one surface, permanent"},
        "composite": {"code": "D2385", "description": "Resin-based composite - three surfaces, permanent"},
        "crown": {"code": "D2750", "description": "Crown - porcelain fused to high noble metal"},
        "root canal": {"code": "D3310", "description": "Endodontic therapy - anterior"},
        "rct": {"code": "D3310", "description": "Endodontic therapy - anterior"},
        "extraction": {"code": "D7140", "description": "Extraction, erupted tooth or exposed root"},
        "implant": {"code": "D6010", "description": "Surgical placement of implant body"},
        "bitewing": {"code": "D0274", "description": "Bitewings - 4 radiographs"},
        "panoramic": {"code": "D0330", "description": "Panoramic radiograph"},
        "fluoride": {"code": "D1206", "description": "Topical fluoride varnish"},
        "sealant": {"code": "D1351", "description": "Sealant - per tooth"},
    }

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Predict CDT codes from clinical text."""
        text_lower = text.lower()
        predictions = []

        for procedure, cdt_info in self.CDT_CODES.items():
            if procedure in text_lower:
                # Calculate confidence based on exact match vs partial
                if f" {procedure} " in f" {text_lower} ":
                    confidence = 0.90
                else:
                    confidence = 0.75

                predictions.append(
                    {
                        "code": cdt_info["code"],
                        "description": cdt_info["description"],
                        "procedure": procedure,
                        "confidence": confidence,
                    }
                )

        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return predictions[:10]  # Return top 10
