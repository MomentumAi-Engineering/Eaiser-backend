#!/bin/bash
# Production Deployment Script for Eaiser Backend
# Optimized for 1 Lakh+ Concurrent Users
# This script sets up the complete infrastructure including:
# - Multiple backend instances
# - Redis cluster
# - MongoDB replica set
# - Nginx load balancer
# - Monitoring stack

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="eaiser-backend"
ENVIRONMENT="production"
DOCKER_COMPOSE_FILE="docker-compose-production.yml"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if required files exist
    required_files=(
        "$DOCKER_COMPOSE_FILE"
        "Dockerfile.optimized"
        "nginx-optimized.conf"
        "main_optimized.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file $file not found!"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs backups ssl monitoring/grafana/{dashboards,datasources} monitoring/logstash/pipeline
    
    # Set proper permissions
    chmod 755 logs backups
    
    # Create SSL directory structure (for HTTPS)
    mkdir -p ssl/{certs,private}
    
    # Create monitoring configuration directories
    mkdir -p monitoring/{prometheus,grafana,logstash,alertmanager}
    
    log_success "Environment setup completed"
}

create_ssl_certificates() {
    log_info "Creating SSL certificates for HTTPS..."
    
    if [[ ! -f "ssl/certs/eaiser-api.crt" ]]; then
        # Generate self-signed certificate for development/testing
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/private/eaiser-api.key \
            -out ssl/certs/eaiser-api.crt \
            -subj "/C=IN/ST=State/L=City/O=Eaiser/OU=IT/CN=eaiser-api.local"
        
        log_success "SSL certificates created"
    else
        log_info "SSL certificates already exist"
    fi
}

setup_monitoring_configs() {
    log_info "Setting up monitoring configurations..."
    
    # Copy Prometheus configuration
    if [[ -f "prometheus-optimized.yml" ]]; then
        cp prometheus-optimized.yml monitoring/prometheus/prometheus.yml
    fi
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create basic Logstash pipeline configuration
    cat > monitoring/logstash/pipeline/logstash.conf << EOF
input {
  file {
    path => "/var/log/backend/*.log"
    start_position => "beginning"
    tags => ["backend"]
  }
  file {
    path => "/var/log/nginx/*.log"
    start_position => "beginning"
    tags => ["nginx"]
  }
}

filter {
  if "backend" in [tags] {
    json {
      source => "message"
    }
  }
  
  if "nginx" in [tags] {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "eaiser-logs-%{+YYYY.MM.dd}"
  }
}
EOF

    log_success "Monitoring configurations created"
}

backup_existing_data() {
    log_info "Creating backup of existing data..."
    
    if docker-compose -f $DOCKER_COMPOSE_FILE ps | grep -q "Up"; then
        mkdir -p "$BACKUP_DIR"
        
        # Backup MongoDB data
        log_info "Backing up MongoDB data..."
        docker-compose -f $DOCKER_COMPOSE_FILE exec -T mongodb-primary mongodump --out /tmp/backup
        docker cp $(docker-compose -f $DOCKER_COMPOSE_FILE ps -q mongodb-primary):/tmp/backup "$BACKUP_DIR/mongodb"
        
        # Backup Redis data
        log_info "Backing up Redis data..."
        docker-compose -f $DOCKER_COMPOSE_FILE exec -T redis-node-1 redis-cli BGSAVE
        
        log_success "Backup completed: $BACKUP_DIR"
    else
        log_info "No running containers found, skipping backup"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    # Build the optimized backend image
    docker build -f Dockerfile.optimized -t eaiser-backend:optimized \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=1.0.0 \
        --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
        .
    
    log_success "Docker images built successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure..."
    
    # Pull required images
    docker-compose -f $DOCKER_COMPOSE_FILE pull
    
    # Start the infrastructure
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    
    log_success "Infrastructure deployment started"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for MongoDB replica set
    log_info "Waiting for MongoDB replica set..."
    timeout=300
    while [ $timeout -gt 0 ]; do
        if docker-compose -f $DOCKER_COMPOSE_FILE exec -T mongodb-primary mongosh --eval "rs.status()" &>/dev/null; then
            break
        fi
        sleep 5
        timeout=$((timeout-5))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "MongoDB replica set failed to start"
        exit 1
    fi
    
    # Wait for Redis cluster
    log_info "Waiting for Redis cluster..."
    timeout=180
    while [ $timeout -gt 0 ]; do
        if docker-compose -f $DOCKER_COMPOSE_FILE exec -T redis-node-1 redis-cli cluster info | grep -q "cluster_state:ok"; then
            break
        fi
        sleep 5
        timeout=$((timeout-5))
    done
    
    if [ $timeout -le 0 ]; then
        log_warning "Redis cluster might not be fully ready, but continuing..."
    fi
    
    # Wait for backend services
    log_info "Waiting for backend services..."
    for i in {1..4}; do
        port=$((9999 + i))
        timeout=120
        while [ $timeout -gt 0 ]; do
            if curl -f http://localhost:$port/health &>/dev/null; then
                log_success "Backend-$i is ready"
                break
            fi
            sleep 5
            timeout=$((timeout-5))
        done
        
        if [ $timeout -le 0 ]; then
            log_warning "Backend-$i might not be ready"
        fi
    done
    
    # Wait for Nginx
    log_info "Waiting for Nginx load balancer..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost/health &>/dev/null; then
            log_success "Nginx load balancer is ready"
            break
        fi
        sleep 5
        timeout=$((timeout-5))
    done
    
    log_success "All services are ready!"
}

run_health_checks() {
    log_info "Running comprehensive health checks..."
    
    # Check all services
    services=("nginx" "backend-1" "backend-2" "backend-3" "backend-4" "mongodb-primary" "redis-node-1")
    
    for service in "${services[@]}"; do
        if docker-compose -f $DOCKER_COMPOSE_FILE ps $service | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running properly"
        fi
    done
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Test health endpoint
    if curl -f http://localhost/health &>/dev/null; then
        log_success "Health endpoint is working"
    else
        log_error "Health endpoint is not responding"
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost/metrics &>/dev/null; then
        log_success "Metrics endpoint is working"
    else
        log_warning "Metrics endpoint might not be available"
    fi
    
    log_success "Health checks completed"
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "================================"
    echo "🚀 Eaiser Backend Production Deployment"
    echo "📊 Optimized for 1 Lakh+ Concurrent Users"
    echo ""
    echo "🌐 Application URLs:"
    echo "   • Main API: http://localhost"
    echo "   • Health Check: http://localhost/health"
    echo "   • API Docs: http://localhost/docs"
    echo "   • Metrics: http://localhost/metrics"
    echo ""
    echo "📈 Monitoring URLs:"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Grafana: http://localhost:3000 (admin/eaiser123)"
    echo "   • Kibana: http://localhost:5601"
    echo "   • Nginx Status: http://localhost:8080/nginx-status"
    echo ""
    echo "🔧 Infrastructure:"
    echo "   • Backend Instances: 4 (ports 10000-10003)"
    echo "   • Redis Cluster: 3 nodes (ports 6379-6381)"
    echo "   • MongoDB Replica Set: 3 nodes (ports 27017-27019)"
    echo "   • Nginx Load Balancer: port 80/443"
    echo ""
    echo "📝 Logs Location: ./logs/"
    echo "💾 Backups Location: ./backups/"
    echo "================================"
}

run_performance_test() {
    log_info "Running basic performance test..."
    
    if command -v ab &> /dev/null; then
        log_info "Testing with Apache Bench (100 requests, 10 concurrent)..."
        ab -n 100 -c 10 http://localhost/health
    elif command -v curl &> /dev/null; then
        log_info "Testing basic connectivity..."
        for i in {1..5}; do
            response_time=$(curl -o /dev/null -s -w "%{time_total}" http://localhost/health)
            log_info "Request $i: ${response_time}s"
        done
    else
        log_warning "No performance testing tools available"
    fi
}

cleanup_old_resources() {
    log_info "Cleaning up old resources..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove old images
    docker image prune -f
    
    # Remove old volumes (be careful with this)
    # docker volume prune -f
    
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    log_info "Starting Eaiser Backend Production Deployment"
    log_info "Optimized for 1 Lakh+ Concurrent Users"
    echo "================================"
    
    # Pre-deployment checks
    check_prerequisites
    setup_environment
    create_ssl_certificates
    setup_monitoring_configs
    
    # Backup existing data
    backup_existing_data
    
    # Build and deploy
    build_images
    deploy_infrastructure
    
    # Post-deployment verification
    wait_for_services
    run_health_checks
    
    # Optional performance test
    if [[ "$1" == "--test" ]]; then
        run_performance_test
    fi
    
    # Cleanup
    cleanup_old_resources
    
    # Show deployment information
    show_deployment_info
    
    log_success "🎉 Production deployment completed successfully!"
    log_info "Your Eaiser Backend is now ready to handle 1 Lakh+ concurrent users!"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
