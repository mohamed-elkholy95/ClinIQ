"""Clinical vital signs extraction module.

Provides structured extraction of vital signs from clinical free text,
including blood pressure, heart rate, temperature, respiratory rate, SpO2,
weight, height, BMI, and pain scale readings.  The rule-based engine uses
compiled regex patterns with unit normalisation, range validation, and
clinical interpretation (normal / abnormal / critical).
"""
