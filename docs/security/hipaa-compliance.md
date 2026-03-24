# ClinIQ HIPAA Compliance Architecture

This document describes ClinIQ's security architecture designed for handling Protected Health Information (PHI) in compliance with the Health Insurance Portability and Accountability Act (HIPAA) Security Rule.

---

## Data Flow with Encryption Boundaries

```
                    ENCRYPTION BOUNDARY
  +---------------------------------------------------------+
  |                                                         |
  |   Client                    ClinIQ Platform             |
  |   +--------+    TLS 1.3    +-----------+                |
  |   | Browser +----------->>-+ Nginx     |                |
  |   | SDK     |   (HTTPS)    | Ingress   |                |
  |   +--------+               +-----+-----+                |
  |                                  |                      |
  |                          +-------+-------+              |
  |                          |  FastAPI API  |              |
  |                          |  (In-memory   |              |
  |                          |   processing) |              |
  |                          +---+---+---+---+              |
  |                              |   |   |                  |
  |                    +---------+   |   +---------+        |
  |                    |             |             |         |
  |              +-----+-----+ +----+----+ +------+-----+  |
  |              | PostgreSQL | |  Redis  | |   MinIO    |  |
  |              | (AES-256   | | (In-mem | | (SSE-S3    |  |
  |              |  at rest)  | |  only)  | |  at rest)  |  |
  |              +------------+ +---------+ +------------+  |
  |                                                         |
  +---------------------------------------------------------+

  Legend:
  >>  = TLS 1.3 encrypted channel
  AES = AES-256 encryption at rest
  SSE = Server-Side Encryption
```

### Encryption Specifications

| Layer | Encryption | Standard |
|-------|-----------|----------|
| **In transit** | TLS 1.3 (minimum TLS 1.2) | All HTTP traffic, database connections, Redis connections |
| **At rest (PostgreSQL)** | AES-256 (pgcrypto or disk-level) | All PHI columns, audit logs |
| **At rest (MinIO)** | SSE-S3 (AES-256) | Uploaded documents, model artifacts |
| **At rest (Redis)** | Not persisted (cache only) | PHI is not stored in Redis; only cache keys and task metadata |
| **Secrets** | Pydantic SecretStr + env vars | Database passwords, API keys, JWT signing keys |

---

## PHI Handling Procedures

### Data Classification

| Classification | Examples | Handling |
|---------------|----------|----------|
| **PHI** | Patient names, DOB, MRN, SSN, clinical notes | Encrypted at rest, access-controlled, audit-logged |
| **De-identified** | Aggregate statistics, model metrics | Standard handling |
| **System Data** | Configuration, logs (without PHI), metrics | Standard handling |

### PHI Lifecycle

```
[1] INGESTION          Clinical text received via HTTPS POST
                       Document hash computed (SHA-256)
                       Original text held in memory only during processing

[2] PROCESSING         ML pipeline processes text in-memory
                       No PHI written to disk during inference
                       SHAP explanations reference text spans, not full text

[3] STORAGE            If store_result=true:
  (Optional)             - Analysis results stored in PostgreSQL (encrypted)
                         - Document content NOT stored by default
                         - Only structured outputs (entities, codes, scores)
                       Uploaded documents stored in MinIO (SSE-S3)

[4] AUDIT              Audit log records:
                         - Document hash (not content)
                         - User/API key identifier
                         - Action performed
                         - Timestamp, IP address, response time
                       Audit logs are append-only and retained per policy

[5] RESPONSE           Results returned via HTTPS
                       In-memory text references released after response
                       No PHI in application logs (structured logging)

[6] DELETION           Configurable retention periods per data class
                       Cryptographic deletion via key rotation
                       MinIO lifecycle policies for document expiry
```

### Minimum Necessary Standard

ClinIQ enforces the HIPAA Minimum Necessary standard:

- API responses contain only the requested analysis results, not the full input text
- The `config` object allows clients to request only the pipeline stages they need
- Batch results include only the requested fields
- Model metadata endpoints never expose training data or patient information

---

## Access Control Matrix

### Role-Based Access Control (RBAC)

| Role | Analyze | NER | ICD | Summary | Risk | Batch | Models | Admin |
|------|---------|-----|-----|---------|------|-------|--------|-------|
| **admin** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **clinician** | Yes | Yes | Yes | Yes | Yes | Yes | Read | No |
| **coder** | No | No | Yes | No | No | Yes (ICD only) | Read | No |
| **researcher** | Yes (de-id) | Yes (de-id) | Yes (de-id) | Yes (de-id) | No | Yes (de-id) | Read | No |
| **api_readonly** | No | No | No | No | No | No | Read | No |

### Authentication Mechanisms

| Mechanism | Use Case | Implementation |
|-----------|----------|----------------|
| **JWT Bearer Token** | Interactive user sessions | HS256, 30-min expiry, bcrypt password hash |
| **API Key** | Programmatic access, SDKs | `cliniq_` prefix + 32-byte urlsafe token, bcrypt hash stored |
| **Service-to-service** | Internal microservices | Shared secret or mTLS (future) |

### API Key Management

- API keys are generated via `POST /api/v1/auth/api-keys` (requires JWT authentication)
- Plaintext key is returned once and never stored
- Bcrypt hash is stored in PostgreSQL
- Keys can be revoked by the issuing user or an admin
- Key rotation: new key issued, old key invalidated after grace period

---

## Audit Logging Specification

### Audit Log Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique audit log entry ID |
| `timestamp` | datetime (UTC) | Event timestamp |
| `action` | string | Action performed (analyze, ner, icd_predict, etc.) |
| `resource_type` | string | Resource category (analysis, model, user, api_key) |
| `user_id` | UUID (nullable) | Authenticated user ID |
| `api_key_prefix` | string (nullable) | First 8 characters of the API key used |
| `document_hash` | string | SHA-256 hash of input document |
| `ip_address` | string | Client IP (X-Forwarded-For aware) |
| `user_agent` | string | Client User-Agent header |
| `status_code` | int | HTTP response status code |
| `response_time_ms` | int | Request processing time |
| `metadata` | JSONB | Additional context (error details, config used) |

### Logged Events

| Event Category | Events |
|---------------|--------|
| **Authentication** | Login success, login failure, token refresh, API key creation, API key revocation |
| **Authorization** | Access denied, role change, permission escalation |
| **Data Access** | Document analysis, NER extraction, ICD prediction, summary generation, risk scoring |
| **Batch Operations** | Batch submission, batch completion, batch failure |
| **Model Operations** | Model load, model swap, model error |
| **Administrative** | User creation, user modification, configuration change, system health check |

### Audit Log Properties

- **Append-only**: Audit logs cannot be modified or deleted through the application
- **Tamper detection**: Log entries include a running hash chain (previous entry hash)
- **Non-blocking**: Audit logging failures do not block the primary request/response
- **Structured**: JSON format for machine-parseable analysis
- **No PHI**: Logs contain document hashes and metadata, never clinical text content

---

## Data Retention Policies

| Data Category | Retention Period | Deletion Method |
|---------------|-----------------|-----------------|
| Audit logs | 7 years | Archived to cold storage after 1 year, deleted after 7 |
| Analysis results | 90 days (configurable) | Soft delete + hard purge after grace period |
| Uploaded documents | 30 days (configurable) | MinIO lifecycle policy (automatic expiry) |
| User accounts | Active + 1 year after deactivation | Anonymized, then purged |
| API keys (revoked) | 90 days after revocation | Hard delete |
| Redis cache | 1 hour TTL (configurable) | Automatic expiry |
| Model artifacts | Indefinite (versioned) | Manual cleanup of deprecated versions |
| Application logs | 30 days | Log rotation (logrotate or container runtime) |

### Data Destruction

When data is deleted:

1. **Soft delete**: Record marked as deleted, excluded from queries
2. **Grace period**: 30 days for accidental deletion recovery
3. **Hard delete**: Record permanently removed from database
4. **Backup purge**: Corresponding entries removed from backups on next rotation
5. **Verification**: Deletion confirmed via audit log and spot check

---

## Incident Response Plan Template

### Severity Classification

| Level | Description | Examples | Response Time |
|-------|-------------|----------|---------------|
| **P1 - Critical** | Active breach of PHI, data exfiltration | Unauthorized access to patient data, database compromise | Immediate (<15 min) |
| **P2 - High** | Potential breach, service compromise | Suspicious access patterns, unpatched vulnerability | <1 hour |
| **P3 - Medium** | Security concern, no confirmed impact | Failed authentication spike, misconfiguration | <4 hours |
| **P4 - Low** | Minor finding, hardening opportunity | Missing header, log verbosity issue | <24 hours |

### Response Procedure

**Phase 1: Detection and Triage (0-15 minutes)**

1. Alert received via monitoring (Grafana), log analysis, or manual report
2. On-call engineer assesses severity using classification matrix above
3. If P1/P2: activate incident response team, begin containment
4. Create incident ticket with timestamp, initial assessment, and severity

**Phase 2: Containment (15-60 minutes)**

1. Isolate affected systems (network segmentation, service shutdown)
2. Revoke compromised credentials (API keys, JWT signing key rotation)
3. Preserve evidence (snapshot logs, database state, container images)
4. Enable enhanced logging on affected systems

**Phase 3: Investigation (1-24 hours)**

1. Review audit logs for scope of access
2. Analyze affected data (document hashes in audit log identify accessed records)
3. Determine root cause (vulnerability, misconfiguration, insider threat)
4. Document timeline of events

**Phase 4: Notification (within 60 days per HIPAA)**

If PHI breach is confirmed affecting 500+ individuals:
1. Notify HHS Office for Civil Rights via breach portal
2. Notify affected individuals in writing
3. Notify prominent media outlets (if 500+ in a single state)

For breaches affecting fewer than 500 individuals:
1. Notify affected individuals within 60 days
2. Log in annual breach report to HHS

**Phase 5: Recovery and Remediation**

1. Patch vulnerability or fix misconfiguration
2. Restore services from known-good state
3. Rotate all credentials (database, Redis, MinIO, JWT keys)
4. Verify containment with penetration test
5. Update monitoring rules to detect similar incidents

**Phase 6: Post-Incident Review**

1. Conduct blameless post-mortem within 5 business days
2. Document lessons learned and remediation actions
3. Update incident response procedures based on findings
4. Schedule follow-up review at 30 days

### Contact List Template

| Role | Name | Contact | Backup |
|------|------|---------|--------|
| Incident Commander | TBD | TBD | TBD |
| Security Lead | TBD | TBD | TBD |
| Engineering Lead | TBD | TBD | TBD |
| Privacy Officer | TBD | TBD | TBD |
| Legal Counsel | TBD | TBD | TBD |
| Communications | TBD | TBD | TBD |

---

## Technical Security Controls

### Network Security

- All external traffic encrypted via TLS 1.3 (minimum TLS 1.2)
- Internal Docker network isolation (`cliniq-prod` bridge network)
- Kubernetes NetworkPolicy restricting pod-to-pod communication
- Ingress rate limiting: 30 RPS with 5x burst multiplier
- Security headers: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy, Content-Security-Policy

### Application Security

- Input validation via Pydantic v2 (max document length: 100,000 chars)
- SQL injection prevention via SQLAlchemy parameterized queries
- CORS restricted to configured origins only
- Rate limiting per API key / IP address
- Request size limits (50 MB via Nginx)
- No PHI in application logs (structured logging with field filtering)

### Infrastructure Security

- Container images scanned with Trivy (CI/CD pipeline)
- Python dependencies scanned with Bandit
- Non-root container execution
- Read-only model volume mounts in production
- Secret management via environment variables (Kubernetes Secrets in K8s)
- Database connection pooling with max overflow limits

### Monitoring and Detection

- Prometheus metrics for anomaly detection (request rates, error rates, latency)
- Audit log analysis for access pattern anomalies
- Failed authentication alerting (threshold: 10 failures in 5 minutes)
- Data drift detection for model behavior monitoring
- SHAP explainability for model output auditing

---

## Compliance Checklist

### HIPAA Security Rule Requirements

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Access Control (164.312(a)(1)) | Implemented | JWT + API Key authentication, RBAC |
| Audit Controls (164.312(b)) | Implemented | Append-only audit log with hash chain |
| Integrity Controls (164.312(c)(1)) | Implemented | TLS in transit, AES-256 at rest |
| Transmission Security (164.312(e)(1)) | Implemented | TLS 1.3 for all external traffic |
| Person Authentication (164.312(d)) | Implemented | bcrypt passwords, unique API keys |
| Automatic Logoff (164.312(a)(2)(iii)) | Implemented | 30-min JWT expiry |
| Encryption (164.312(a)(2)(iv)) | Implemented | AES-256 at rest, TLS 1.3 in transit |
| Emergency Access (164.312(a)(2)(ii)) | Planned | Break-glass procedure (documented) |
| Unique User Identification (164.312(a)(2)(i)) | Implemented | UUID-based user and API key tracking |

### Administrative Safeguards

| Requirement | Status | Notes |
|-------------|--------|-------|
| Risk Analysis | Planned | Annual risk assessment procedure |
| Workforce Training | Planned | Security awareness program |
| Contingency Plan | Documented | Backup and recovery procedures above |
| Business Associate Agreements | Required | Template needed for each BAA partner |
| Incident Response | Documented | See incident response plan above |

This document should be reviewed and updated quarterly, or whenever significant architectural changes are made.
