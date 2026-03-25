"""Clinical document type classification module.

Classifies clinical notes into document types such as discharge summaries,
progress notes, radiology reports, operative notes, and more.  Uses a
combination of rule-based section/keyword detection and optional transformer
classification with automatic fallback.
"""
