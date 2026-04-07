# ClinIQ Documentation

This directory contains the project documentation for ClinIQ.

Use this folder as the primary entry point for architecture, deployment, security, API, and model-level reference material.

## Start Here

- [Architecture](architecture.md): system design, core components, data flow, and major decisions
- [API Reference](api/api-reference.md): endpoint-level reference for the backend API
- [Local Setup](deployment/local-setup.md): local development environment and service startup
- [Production Guide](deployment/production-guide.md): deployment patterns for Docker Compose and Kubernetes
- [HIPAA Compliance Architecture](security/hipaa-compliance.md): security controls, PHI handling, audit expectations, and compliance posture

## ML Documentation

The `ml/` directory contains model cards and evaluation notes for the clinical NLP modules, including:

- NER
- ICD-10 prediction
- summarization
- risk scoring
- de-identification
- relations
- temporal extraction
- medications
- vitals
- document classification
- conversation memory
- other task-specific components

Browse the folder here: [ML documentation](ml/)

## Recommended Reading Order

If you are new to the repository, read in this order:

1. [README.md](../README.md)
2. [Architecture](architecture.md)
3. [Local Setup](deployment/local-setup.md)
4. [API Reference](api/api-reference.md)
5. [HIPAA Compliance Architecture](security/hipaa-compliance.md)

## Public Repo Reminder

This repository is public.

- Do not add secrets or real deployment credentials to docs.
- Do not include PHI, real patient examples, or internal-only operational detail.
- Use placeholders and synthetic examples only.

Contributor and agent rules are documented in [AGENTS.md](../AGENTS.md).
