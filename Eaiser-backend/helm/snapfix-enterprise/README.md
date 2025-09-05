# 🚀 SnapFix Enterprise Helm Chart

[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.25%2B-blue.svg)](https://kubernetes.io/)
[![Helm](https://img.shields.io/badge/Helm-3.8%2B-blue.svg)](https://helm.sh/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](Chart.yaml)

## 📋 Overview

**SnapFix Enterprise** is a production-ready Helm chart designed for deploying high-performance FastAPI applications capable of handling **100,000+ concurrent users**. This chart provides enterprise-grade features including auto-scaling, comprehensive monitoring, advanced security, and multi-environment support.

### 🎯 Key Features

- **🔄 Auto-Scaling**: HPA, VPA, and KEDA for intelligent scaling
- **📊 Monitoring**: Prometheus, Grafana, and Jaeger integration
- **🔒 Security**: Network policies, RBAC, and certificate management
- **🗄️ Multi-Database**: PostgreSQL, MongoDB, and Redis support
- **📨 Message Queue**: RabbitMQ with Celery workers
- **🌐 Ingress**: High-performance Nginx with SSL termination
- **⚡ Performance**: Optimized for high-concurrency workloads
- **🔄 GitOps**: ArgoCD integration for continuous deployment
- **💾 Backup**: Velero for disaster recovery
- **🕸️ Service Mesh**: Optional Istio support

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "External"
        U[Users] --> LB[Load Balancer]
        CDN[CloudFlare CDN] --> LB
    end
    
    subgraph "Kubernetes Cluster"
        LB --> IG[Nginx Ingress]
        IG --> APP[FastAPI Pods]
        
        subgraph "Application Layer"
            APP --> |Read/Write| PG[(PostgreSQL)]
            APP --> |Cache| RD[(Redis Cluster)]
            APP --> |Documents| MG[(MongoDB)]
            APP --> |Tasks| RMQ[RabbitMQ]
        end
        
        subgraph "Worker Layer"
            RMQ --> CW[Celery Workers]
            CW --> CB[Celery Beat]
            CW --> CF[Celery Flower]
        end
        
        subgraph "Monitoring"
            APP --> PR[Prometheus]
            PR --> GR[Grafana]
            APP --> JG[Jaeger]
        end
        
        subgraph "Logging"
            APP --> ES[Elasticsearch]
            ES --> KB[Kibana]
        end
    end
```

## 🚀 Quick Start

### Prerequisites

- Kubernetes 1.25+
- Helm 3.8+
- kubectl configured
- Sufficient cluster resources (see [Resource Requirements](#-resource-requirements))

### 1. Add Helm Repository

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add jetstack https://charts.jetstack.io
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts
helm repo update
```

### 2. Create Namespace

```bash
# Create namespace for your environment
kubectl create namespace snapfix-enterprise-dev
kubectl create namespace snapfix-enterprise-staging
kubectl create namespace snapfix-enterprise-prod
```

### 3. Install Dependencies

```bash
# Install cert-manager (required for TLS)
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Install ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace
```

### 4. Deploy Application

#### Development Environment

```bash
helm install snapfix-dev . \
  --namespace snapfix-enterprise-dev \
  --values values-dev.yaml \
  --create-namespace
```

#### Staging Environment

```bash
helm install snapfix-staging . \
  --namespace snapfix-enterprise-staging \
  --values values-staging.yaml \
  --create-namespace
```

#### Production Environment

```bash
# Create secrets first (see Security section)
kubectl create secret generic snapfix-database-secret-prod \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --namespace snapfix-enterprise-prod

# Deploy production
helm install snapfix-prod . \
  --namespace snapfix-enterprise-prod \
  --values values-production.yaml \
  --create-namespace
```

## 📁 File Structure

```
snapfix-enterprise/
├── Chart.yaml                 # Chart metadata
├── values.yaml               # Default configuration
├── values-dev.yaml           # Development overrides
├── values-staging.yaml       # Staging overrides
├── values-production.yaml    # Production overrides
├── README.md                 # This file
├── templates/
│   ├── deployment.yaml       # Main application deployment
│   ├── service.yaml          # Kubernetes service
│   ├── ingress.yaml          # Ingress configuration
│   ├── hpa.yaml             # Horizontal Pod Autoscaler
│   ├── vpa.yaml             # Vertical Pod Autoscaler
│   ├── pdb.yaml             # Pod Disruption Budget
│   ├── configmap.yaml       # Configuration maps
│   ├── secrets.yaml         # Secret templates
│   ├── rbac.yaml            # RBAC configuration
│   ├── networkpolicy.yaml   # Network policies
│   ├── servicemonitor.yaml  # Prometheus monitoring
│   ├── prometheusrule.yaml  # Alerting rules
│   ├── celery/              # Celery worker deployments
│   │   ├── worker.yaml
│   │   ├── beat.yaml
│   │   └── flower.yaml
│   └── NOTES.txt            # Post-installation notes
└── charts/                   # Dependency charts (auto-generated)
```

## ⚙️ Configuration

### Environment-Specific Values

| Environment | File | Description | Resources |
|-------------|------|-------------|----------|
| **Development** | `values-dev.yaml` | Minimal resources, debug enabled | 2 replicas, 512Mi RAM |
| **Staging** | `values-staging.yaml` | Production-like, moderate resources | 5 replicas, 2Gi RAM |
| **Production** | `values-production.yaml` | Maximum performance and security | 20+ replicas, 8Gi RAM |

### Key Configuration Sections

#### Application Configuration

```yaml
app:
  replicaCount: 20              # Number of pod replicas
  image:
    repository: your-registry.com/snapfix-enterprise
    tag: "v1.0.0"
  resources:
    requests:
      cpu: "1000m"
      memory: "2Gi"
    limits:
      cpu: "4000m"
      memory: "8Gi"
```

#### Auto-Scaling Configuration

```yaml
autoscaling:
  hpa:
    enabled: true
    minReplicas: 15
    maxReplicas: 100
    targetCPUUtilizationPercentage: 60
    customMetrics:
      enabled: true
  vpa:
    enabled: true
    updateMode: "Auto"
  keda:
    enabled: true
```

#### Database Configuration

```yaml
postgresql:
  enabled: true
  architecture: replication
  primary:
    resources:
      requests:
        cpu: "2000m"
        memory: "4Gi"
    persistence:
      size: 500Gi
```

#### Monitoring Configuration

```yaml
monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
kube-prometheus-stack:
  enabled: true
  prometheus:
    prometheusSpec:
      retention: 30d
      storageSpec:
        volumeClaimTemplate:
          spec:
            resources:
              requests:
                storage: 200Gi
```

## 🔒 Security

### Secret Management

Create required secrets before deployment:

```bash
# Database secrets
kubectl create secret generic snapfix-database-secret-prod \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --namespace snapfix-enterprise-prod

# Redis secrets
kubectl create secret generic snapfix-redis-secret-prod \
  --from-literal=redis-url="redis://:password@redis-cluster:6379/0" \
  --namespace snapfix-enterprise-prod

# MongoDB secrets
kubectl create secret generic snapfix-mongodb-secret-prod \
  --from-literal=mongodb-url="mongodb://user:pass@mongodb:27017/snapfix" \
  --namespace snapfix-enterprise-prod

# RabbitMQ secrets
kubectl create secret generic snapfix-rabbitmq-secret-prod \
  --from-literal=rabbitmq-url="amqp://user:pass@rabbitmq:5672/" \
  --namespace snapfix-enterprise-prod

# Security secrets
kubectl create secret generic snapfix-security-secret-prod \
  --from-literal=jwt-secret="your-super-secret-jwt-key" \
  --namespace snapfix-enterprise-prod
```

### Network Policies

Network policies are enabled by default in staging and production:

```yaml
security:
  networkPolicies:
    enabled: true
    defaultDenyIngress: true
```

### TLS/SSL Configuration

Certificates are automatically managed by cert-manager:

```yaml
security:
  certManager:
    enabled: true
    clusterIssuer:
      name: letsencrypt-prod
      email: devops@snapfix.enterprise.com
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The application exposes metrics on `/metrics` endpoint:

- HTTP request metrics
- Database connection pool metrics
- Redis connection metrics
- Custom business metrics

### Grafana Dashboards

Pre-configured dashboards include:

- Application Performance
- Infrastructure Metrics
- Database Performance
- Redis Metrics
- Celery Worker Metrics

### Alerting Rules

Comprehensive alerting rules for:

- High error rates
- Response time degradation
- Resource exhaustion
- Database issues
- Worker queue buildup

### Distributed Tracing

Jaeger integration provides:

- Request tracing across services
- Performance bottleneck identification
- Dependency mapping

## 🔄 Auto-Scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
autoscaling:
  hpa:
    enabled: true
    minReplicas: 15
    maxReplicas: 100
    targetCPUUtilizationPercentage: 60
    targetMemoryUtilizationPercentage: 70
    customMetrics:
      - http_requests_per_second
      - database_connections_active
```

### Vertical Pod Autoscaler (VPA)

```yaml
autoscaling:
  vpa:
    enabled: true
    updateMode: "Auto"
    resourcePolicy:
      containerPolicies:
        - containerName: snapfix-enterprise
          minAllowed:
            cpu: "500m"
            memory: "1Gi"
          maxAllowed:
            cpu: "8000m"
            memory: "16Gi"
```

### KEDA (Event-Driven Autoscaling)

```yaml
autoscaling:
  keda:
    enabled: true
    scaledObjects:
      - name: rabbitmq-scaler
        triggers:
          - type: rabbitmq
            metadata:
              queueName: snapfix-tasks
              queueLength: "10"
```

## 💾 Backup & Disaster Recovery

### Velero Backup

Automated backup schedules:

```yaml
backup:
  velero:
    enabled: true
    schedules:
      daily:
        schedule: "0 2 * * *"
        template:
          ttl: "720h"  # 30 days
      weekly:
        schedule: "0 3 * * 0"
        template:
          ttl: "2016h"  # 12 weeks
```

### Database Backups

- PostgreSQL: Automated pg_dump with WAL archiving
- MongoDB: Replica set with automated snapshots
- Redis: RDB and AOF persistence with replication

## 🔄 GitOps Integration

### ArgoCD Application

```yaml
gitops:
  argocd:
    applications:
      production:
        name: snapfix-enterprise-prod
        source:
          repoURL: https://github.com/your-org/snapfix-enterprise-helm
          targetRevision: main
          helm:
            valueFiles:
              - values.yaml
              - values-production.yaml
        syncPolicy:
          automated:
            prune: true
            selfHeal: true
```

## 📈 Performance Tuning

### Application Optimization

```yaml
app:
  env:
    - name: WORKERS
      value: "8"                    # Gunicorn workers
    - name: MAX_CONNECTIONS
      value: "2000"                 # Max concurrent connections
    - name: KEEP_ALIVE
      value: "65"                   # Keep-alive timeout
    - name: DATABASE_POOL_SIZE
      value: "50"                   # DB connection pool
    - name: REDIS_POOL_SIZE
      value: "100"                  # Redis connection pool
```

### Nginx Ingress Optimization

```yaml
nginx-ingress:
  controller:
    config:
      worker-processes: "auto"
      worker-connections: "65536"
      proxy-buffer-size: "16k"
      proxy-buffers-number: "8"
      use-gzip: "true"
      gzip-level: "6"
```

### Database Optimization

```yaml
postgresql:
  primary:
    configuration: |
      max_connections = 1000
      shared_buffers = 4GB
      effective_cache_size = 12GB
      work_mem = 32MB
      maintenance_work_mem = 1GB
```

## 📊 Resource Requirements

### Minimum Requirements (Development)

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Application | 500m | 1Gi | 2Gi |
| PostgreSQL | 250m | 512Mi | 20Gi |
| Redis | 125m | 256Mi | 5Gi |
| MongoDB | 250m | 512Mi | 20Gi |
| RabbitMQ | 125m | 256Mi | 5Gi |
| **Total** | **1.25** | **2.5Gi** | **52Gi** |

### Recommended Requirements (Production)

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Application (20 pods) | 20 | 40Gi | 40Gi |
| PostgreSQL Cluster | 12 | 24Gi | 1.5Ti |
| Redis Cluster | 6 | 12Gi | 300Gi |
| MongoDB Cluster | 12 | 24Gi | 1Ti |
| RabbitMQ Cluster | 5 | 10Gi | 500Gi |
| Monitoring Stack | 8 | 16Gi | 500Gi |
| **Total** | **63** | **126Gi** | **3.84Ti** |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n snapfix-enterprise-prod

# Check pod logs
kubectl logs -f deployment/snapfix-enterprise -n snapfix-enterprise-prod

# Describe pod for events
kubectl describe pod <pod-name> -n snapfix-enterprise-prod
```

#### 2. Database Connection Issues

```bash
# Check database pod status
kubectl get pods -l app.kubernetes.io/name=postgresql -n snapfix-enterprise-prod

# Test database connectivity
kubectl exec -it deployment/snapfix-enterprise -n snapfix-enterprise-prod -- \
  python -c "import psycopg2; print('DB OK')"
```

#### 3. High Memory Usage

```bash
# Check resource usage
kubectl top pods -n snapfix-enterprise-prod

# Check VPA recommendations
kubectl get vpa -n snapfix-enterprise-prod

# Scale up resources if needed
kubectl patch deployment snapfix-enterprise -n snapfix-enterprise-prod -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"snapfix-enterprise","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

#### 4. Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n snapfix-enterprise-prod

# Check ingress controller logs
kubectl logs -f deployment/ingress-nginx-controller -n ingress-nginx

# Check certificate status
kubectl get certificates -n snapfix-enterprise-prod
```

### Performance Issues

#### 1. Slow Response Times

```bash
# Check application metrics
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Open http://localhost:9090 and query: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Check database performance
kubectl exec -it postgresql-primary-0 -n snapfix-enterprise-prod -- \
  psql -U snapfix_prod -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

#### 2. High CPU Usage

```bash
# Check HPA status
kubectl get hpa -n snapfix-enterprise-prod

# Check if scaling is working
kubectl describe hpa snapfix-enterprise -n snapfix-enterprise-prod

# Manual scaling if needed
kubectl scale deployment snapfix-enterprise --replicas=30 -n snapfix-enterprise-prod
```

### Monitoring Commands

```bash
# Check all resources
kubectl get all -n snapfix-enterprise-prod

# Check resource usage
kubectl top nodes
kubectl top pods -n snapfix-enterprise-prod

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp -n snapfix-enterprise-prod

# Check logs
kubectl logs -f deployment/snapfix-enterprise -n snapfix-enterprise-prod --tail=100

# Port forward to services
kubectl port-forward svc/grafana 3000:80 -n monitoring
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
kubectl port-forward svc/jaeger-query 16686:16686 -n monitoring
```

## 🔧 Customization

### Adding Custom Environment Variables

```yaml
app:
  env:
    - name: CUSTOM_SETTING
      value: "custom-value"
    - name: SECRET_SETTING
      valueFrom:
        secretKeyRef:
          name: custom-secret
          key: secret-key
```

### Adding Custom ConfigMaps

```yaml
configMaps:
  custom-config:
    data:
      config.json: |
        {
          "setting1": "value1",
          "setting2": "value2"
        }
```

### Adding Custom Volumes

```yaml
volumes:
  - name: custom-volume
    configMap:
      name: custom-config

volumesMounts:
  - name: custom-volume
    mountPath: /app/config
    readOnly: true
```

## 📚 Additional Resources

### Documentation

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)

### Best Practices

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Security Best Practices](https://kubernetes.io/docs/concepts/security/)

### Community

- [Kubernetes Slack](https://kubernetes.slack.com/)
- [Helm Community](https://helm.sh/community/)
- [CNCF Slack](https://slack.cncf.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- 📧 Email: devops@snapfix.enterprise.com
- 💬 Slack: #snapfix-enterprise
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/snapfix-enterprise-helm/issues)
- 📖 Wiki: [Project Wiki](https://github.com/your-org/snapfix-enterprise-helm/wiki)

---

**Made with ❤️ by the SnapFix Enterprise Team**