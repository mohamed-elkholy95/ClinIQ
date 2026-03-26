# ClinIQ Production Deployment Guide

This guide covers deploying ClinIQ to production using Docker Compose and Kubernetes, including SSL/TLS configuration, monitoring setup, backup procedures, and scaling considerations.

---

## Docker Compose Production Deployment

### Prerequisites

- Linux server with Docker 24+ and Docker Compose v2+
- Minimum 8 GB RAM, 4 CPU cores
- Domain name with DNS configured
- SSL/TLS certificate (or use Let's Encrypt)

### 1. Prepare Environment

```bash
git clone https://github.com/cliniq/cliniq.git
cd cliniq

# Create production .env file
cat > .env << 'EOF'
# Required - generate with: openssl rand -hex 32
SECRET_KEY=your-secret-key-here

# PostgreSQL
POSTGRES_USER=cliniq
POSTGRES_PASSWORD=strong-password-here
POSTGRES_DB=cliniq

# Redis
REDIS_PASSWORD=strong-redis-password-here

# MinIO
MINIO_ROOT_USER=cliniq-minio
MINIO_ROOT_PASSWORD=strong-minio-password-here

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=strong-grafana-password-here
EOF

chmod 600 .env
```

### 2. Build and Deploy

```bash
# Build production images
docker compose -f docker-compose.prod.yml build

# Start all services
docker compose -f docker-compose.prod.yml up -d

# Verify services
docker compose -f docker-compose.prod.yml ps
```

### 3. Run Database Migrations

```bash
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### 4. Verify Deployment

```bash
# Health check
curl -s http://localhost:8000/api/v1/health | jq .

# Test NER endpoint
curl -s -X POST http://localhost:8000/api/v1/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient takes metformin 1000mg BID"}' | jq .
```

### Production Docker Compose Architecture

The production compose file (`docker-compose.prod.yml`) includes:

| Service | Image | Resources (limits) |
|---------|-------|--------------------|
| `nginx` | nginx:1.25-alpine | 0.5 CPU, 256 MB |
| `frontend` | cliniq/frontend (built) | 0.5 CPU, 256 MB |
| `api` | cliniq/api (built) | 2 CPU, 4 GB |
| `worker` | cliniq/api (Celery) | 2 CPU, 4 GB |
| `postgres` | postgres:16-alpine | 2 CPU, 2 GB |
| `redis` | redis:7-alpine | 1 CPU, 1 GB |
| `minio` | minio/minio | 1 CPU, 1 GB |
| `mlflow` | mlflow:v2.9.0 | 1 CPU, 1 GB |
| `prometheus` | prom/prometheus | 1 CPU, 2 GB |
| `grafana` | grafana/grafana | 1 CPU, 512 MB |
| `node-exporter` | prom/node-exporter | 0.25 CPU, 128 MB |
| `redis-exporter` | redis_exporter | 0.25 CPU, 64 MB |
| `postgres-exporter` | postgres-exporter | 0.25 CPU, 64 MB |

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.27+) with `kubectl` configured
- Ingress controller (nginx-ingress recommended)
- cert-manager (for automated TLS certificates)
- Persistent volume provisioner (for model storage)

### 1. Create Namespace and Secrets

```bash
# Create namespace
kubectl apply -f infra/k8s/namespace.yml

# Create secrets
kubectl create secret generic cliniq-secrets \
  --namespace=cliniq \
  --from-literal=database-url='postgresql+asyncpg://cliniq:PASSWORD@postgres:5432/cliniq' \
  --from-literal=redis-url='redis://:PASSWORD@redis:6379/0' \
  --from-literal=secret-key="$(openssl rand -hex 32)" \
  --from-literal=minio-access-key='cliniq-minio' \
  --from-literal=minio-secret-key='MINIO-PASSWORD'
```

### 2. Deploy Application

```bash
# Storage and configuration
kubectl apply -f infra/k8s/storage.yml
kubectl apply -f infra/k8s/configmap.yml

# Data layer
kubectl apply -f infra/k8s/postgres-deployment.yml
kubectl apply -f infra/k8s/redis-deployment.yml

# Application
kubectl apply -f infra/k8s/api-deployment.yml
kubectl apply -f infra/k8s/api-service.yml
kubectl apply -f infra/k8s/frontend-deployment.yml
kubectl apply -f infra/k8s/worker-deployment.yml

# Monitoring
kubectl apply -f infra/k8s/monitoring-deployment.yml

# Ingress
kubectl apply -f infra/k8s/ingress.yml

# Verify
kubectl get pods -n cliniq
kubectl get svc -n cliniq
kubectl get ingress -n cliniq
kubectl get hpa -n cliniq
```

### 3. Kubernetes Resource Configuration

The API deployment (`infra/k8s/api-deployment.yml`) includes:

**Resource Requests/Limits:**
- Requests: 500m CPU, 512 Mi memory
- Limits: 2 CPU, 4 Gi memory

**Health Probes:**
- Liveness: `GET /api/v1/health` every 15s (fail after 3)
- Readiness: `GET /api/v1/health` every 10s (fail after 3)
- Startup: `GET /api/v1/health` every 5s (fail after 12, allowing 60s startup)

**Horizontal Pod Autoscaler:**
- Min replicas: 2
- Max replicas: 10
- Scale-up trigger: 70% CPU or 80% memory utilization
- Scale-up: +2 pods per 60s
- Scale-down: -1 pod per 120s (300s stabilization window)

### 4. Ingress Configuration

The ingress (`infra/k8s/ingress.yml`) provides:

- TLS termination via cert-manager (Let's Encrypt)
- Path-based routing: `/api/*` to API service, `/` to frontend
- Rate limiting: 30 RPS with 5x burst multiplier
- Security headers: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy
- Proxy body size limit: 50 MB
- Proxy timeouts: 120s read, 120s send

---

## SSL/TLS Configuration

### Docker Compose (Nginx)

Place certificates in the `ssl_certs` volume:

```bash
# Using Let's Encrypt with certbot
sudo certbot certonly --standalone -d cliniq.example.com

# Copy certificates
docker compose -f docker-compose.prod.yml exec nginx sh -c \
  "mkdir -p /etc/nginx/ssl"

# Update nginx.conf to reference:
#   ssl_certificate /etc/nginx/ssl/fullchain.pem;
#   ssl_certificate_key /etc/nginx/ssl/privkey.pem;
```

### Kubernetes (cert-manager)

The ingress is pre-configured for cert-manager:

```yaml
annotations:
  cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - cliniq.example.com
      secretName: cliniq-tls
```

Install cert-manager and create the ClusterIssuer:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

# Create Let's Encrypt issuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@cliniq.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

---

## Monitoring Setup

### Prometheus

Prometheus is pre-configured to scrape:

| Target | Endpoint | Interval |
|--------|----------|----------|
| ClinIQ API | `:8000/metrics` | 15s |
| PostgreSQL Exporter | `:9187/metrics` | 15s |
| Redis Exporter | `:9121/metrics` | 15s |
| Node Exporter | `:9100/metrics` | 15s |

Access Prometheus at http://localhost:9090 (Docker Compose) or via port-forward in Kubernetes.

### Grafana

Grafana is pre-provisioned with:

- **Data source**: Prometheus (auto-configured)
- **Dashboard**: ClinIQ Platform Dashboard (`infra/grafana/provisioning/dashboards/cliniq-dashboard.json`)

Key panels:
- Request rate and latency (p50, p95, p99)
- Model inference time by model type
- Error rate by endpoint
- Database connection pool utilization
- Redis cache hit rate
- System resources (CPU, memory, disk)

Access Grafana at http://localhost:3001.

### Alerting

Configure Grafana alerting for:

| Alert | Condition | Severity |
|-------|-----------|----------|
| High error rate | >5% 5xx errors over 5 min | Critical |
| Slow inference | p95 latency >5s over 5 min | Warning |
| Database connection saturation | Pool usage >90% | Warning |
| Disk space low | <10% free | Critical |
| Redis memory high | >80% maxmemory | Warning |

---

## Backup and Recovery

### PostgreSQL Backup

**Automated daily backups:**

```bash
# Create backup script
cat > /opt/cliniq/backup.sh << 'SCRIPT'
#!/bin/bash
BACKUP_DIR="/opt/cliniq/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# Dump database
docker compose -f /opt/cliniq/docker-compose.prod.yml exec -T postgres \
  pg_dump -U cliniq -Fc cliniq > "$BACKUP_DIR/cliniq_${TIMESTAMP}.dump"

# Compress
gzip "$BACKUP_DIR/cliniq_${TIMESTAMP}.dump"

# Remove old backups
find "$BACKUP_DIR" -name "*.dump.gz" -mtime +${RETENTION_DAYS} -delete

echo "Backup completed: cliniq_${TIMESTAMP}.dump.gz"
SCRIPT

chmod +x /opt/cliniq/backup.sh

# Add to crontab (daily at 2 AM)
echo "0 2 * * * /opt/cliniq/backup.sh >> /var/log/cliniq-backup.log 2>&1" | crontab -
```

**Restore from backup:**

```bash
gunzip cliniq_20260324_020000.dump.gz

docker compose -f docker-compose.prod.yml exec -T postgres \
  pg_restore -U cliniq -d cliniq --clean --if-exists < cliniq_20260324_020000.dump
```

### Redis Backup

Redis is configured with AOF persistence. For manual snapshots:

```bash
docker compose -f docker-compose.prod.yml exec redis redis-cli -a "$REDIS_PASSWORD" BGSAVE
```

### MinIO Backup

Use MinIO Client (`mc`) to mirror data:

```bash
mc alias set cliniq http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc mirror cliniq/cliniq /opt/cliniq/backups/minio/
```

### Model Artifacts

Model files in `./models/` should be:
- Version controlled in a separate Git LFS repository or MLflow artifact store
- Backed up to S3/MinIO as part of the MinIO backup
- Immutable: never overwrite a model file; create a new version instead

---

## Scaling Considerations

### Horizontal Scaling

| Component | Scaling Strategy | Notes |
|-----------|-----------------|-------|
| API | Add replicas | Stateless; scale based on CPU/request rate |
| Celery Workers | Add workers | Increase `--concurrency` or add containers |
| PostgreSQL | Read replicas | Use connection pooler (PgBouncer) for many connections |
| Redis | Redis Cluster or Sentinel | For cache; broker can stay single-node |
| Frontend | CDN + replicas | Static assets; minimal compute |

### Vertical Scaling

| Component | CPU | Memory | When to Scale |
|-----------|-----|--------|---------------|
| API | 2-4 cores | 4-8 GB | Transformer inference; concurrent requests |
| Worker | 2-4 cores | 4-8 GB | Batch processing; large documents |
| PostgreSQL | 2-4 cores | 2-4 GB | Complex queries; many concurrent connections |
| Redis | 1-2 cores | 1-2 GB | Large cache working set; high throughput |

### GPU Deployment

For transformer model inference:

```yaml
# Kubernetes GPU deployment
resources:
  limits:
    nvidia.com/gpu: 1
    cpu: "2"
    memory: 8Gi
```

Docker Compose (NVIDIA Container Toolkit):

```yaml
api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Performance Benchmarks

Tested on 4-core CPU, 8 GB RAM (no GPU):

| Endpoint | p50 Latency | p95 Latency | Throughput |
|----------|-------------|-------------|------------|
| NER (rule-based) | 8 ms | 15 ms | 120 req/s |
| NER (transformer) | 180 ms | 350 ms | 5 req/s |
| ICD-10 (sklearn) | 25 ms | 45 ms | 40 req/s |
| ICD-10 (transformer) | 220 ms | 450 ms | 4 req/s |
| Summarize (extractive) | 15 ms | 30 ms | 65 req/s |
| Summarize (abstractive) | 850 ms | 1800 ms | 1 req/s |
| Risk score | 5 ms | 12 ms | 200 req/s |
| Full pipeline | 60 ms | 120 ms | 15 req/s |

---

## Operational Procedures

### Rolling Updates

Docker Compose:

```bash
docker compose -f docker-compose.prod.yml build api
docker compose -f docker-compose.prod.yml up -d --no-deps api
```

Kubernetes:

```bash
kubectl set image deployment/cliniq-api api=cliniq/api:v1.2.0 -n cliniq
kubectl rollout status deployment/cliniq-api -n cliniq
```

### Log Access

```bash
# Docker Compose
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f worker

# Kubernetes
kubectl logs -f deployment/cliniq-api -n cliniq
kubectl logs -f deployment/cliniq-api -n cliniq --previous  # Previous crash
```

### Health Checks

```bash
# Quick health
curl -s http://localhost:8000/api/v1/health | jq .

# Readiness (includes dependency checks)
curl -s http://localhost:8000/api/v1/health/ready | jq .

# Liveness (basic process check)
curl -s http://localhost:8000/api/v1/health/live | jq .
```
