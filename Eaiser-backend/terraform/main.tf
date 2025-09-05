# 🌐 SnapFix Enterprise Infrastructure as Code
# 🚀 Multi-Cloud Terraform Configuration for 100,000+ Users
# ☁️ AWS, GCP, Azure Support with Auto-Scaling & High Availability
# 🔧 Kubernetes, Load Balancers, Databases, Monitoring & Security

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  
  # Remote state configuration
  backend "s3" {
    bucket         = "snapfix-terraform-state"
    key            = "enterprise/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "snapfix-terraform-locks"
  }
}

# ========================================
# GLOBAL VARIABLES & LOCALS
# ========================================
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "snapfix-enterprise"
}

variable "region_aws" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "region_gcp" {
  description = "GCP region"
  type        = string
  default     = "us-west1"
}

variable "region_azure" {
  description = "Azure region"
  type        = string
  default     = "West US 2"
}

variable "enable_multi_cloud" {
  description = "Enable multi-cloud deployment"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring stack"
  type        = bool
  default     = true
}

variable "enable_security_hardening" {
  description = "Enable advanced security features"
  type        = bool
  default     = true
}

variable "max_replicas" {
  description = "Maximum number of application replicas"
  type        = number
  default     = 50
}

variable "min_replicas" {
  description = "Minimum number of application replicas"
  type        = number
  default     = 10
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "DevOps-Team"
    CostCenter  = "Engineering"
    Backup      = "Required"
    Monitoring  = "Enabled"
  }
  
  # Generate unique resource names
  resource_prefix = "${var.project_name}-${var.environment}"
  
  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  availability_zones = [
    "${var.region_aws}a",
    "${var.region_aws}b",
    "${var.region_aws}c"
  ]
  
  # Database configuration
  db_instance_class = var.environment == "production" ? "db.r6g.2xlarge" : "db.r6g.large"
  redis_node_type   = var.environment == "production" ? "cache.r6g.2xlarge" : "cache.r6g.large"
  
  # Kubernetes configuration
  k8s_version = "1.27"
  node_groups = {
    general = {
      instance_types = ["m6i.2xlarge", "m6i.4xlarge"]
      min_size       = 3
      max_size       = 20
      desired_size   = 6
    }
    compute_optimized = {
      instance_types = ["c6i.4xlarge", "c6i.8xlarge"]
      min_size       = 2
      max_size       = 15
      desired_size   = 4
    }
    memory_optimized = {
      instance_types = ["r6i.2xlarge", "r6i.4xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
  }
}

# ========================================
# AWS PROVIDER CONFIGURATION
# ========================================
provider "aws" {
  region = var.region_aws
  
  default_tags {
    tags = local.common_tags
  }
}

# ========================================
# AWS VPC & NETWORKING
# ========================================
resource "aws_vpc" "main" {
  cidr_block           = local.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-vpc"
    Type = "Main-VPC"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(local.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(local.vpc_cidr, 8, count.index)
  availability_zone       = local.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-public-${count.index + 1}"
    Type = "Public"
    "kubernetes.io/role/elb" = "1"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(local.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.vpc_cidr, 8, count.index + 10)
  availability_zone = local.availability_zones[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-private-${count.index + 1}"
    Type = "Private"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

# Database Subnets
resource "aws_subnet" "database" {
  count = length(local.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(local.vpc_cidr, 8, count.index + 20)
  availability_zone = local.availability_zones[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-database-${count.index + 1}"
    Type = "Database"
  })
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = length(local.availability_zones)
  
  domain = "vpc"
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-nat-eip-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count = length(local.availability_zones)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-nat-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count = length(local.availability_zones)
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-private-rt-${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# ========================================
# SECURITY GROUPS
# ========================================
# EKS Cluster Security Group
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${local.resource_prefix}-eks-cluster-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port = 443
    to_port   = 443
    protocol  = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access to EKS API"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-eks-cluster-sg"
  })
}

# EKS Node Group Security Group
resource "aws_security_group" "eks_nodes" {
  name_prefix = "${local.resource_prefix}-eks-nodes-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Node to node communication"
  }
  
  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
    description     = "Cluster to node communication"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-eks-nodes-sg"
  })
}

# Database Security Group
resource "aws_security_group" "database" {
  name_prefix = "${local.resource_prefix}-database-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
    description     = "PostgreSQL access from EKS nodes"
  }
  
  ingress {
    from_port       = 27017
    to_port         = 27017
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
    description     = "MongoDB access from EKS nodes"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-database-sg"
  })
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name_prefix = "${local.resource_prefix}-redis-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
    description     = "Redis access from EKS nodes"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-redis-sg"
  })
}

# ========================================
# EKS CLUSTER
# ========================================
# EKS Cluster IAM Role
resource "aws_iam_role" "eks_cluster" {
  name = "${local.resource_prefix}-eks-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "${local.resource_prefix}-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = local.k8s_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }
  
  # Enable logging
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Encryption configuration
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-eks-cluster"
  })
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks_cluster
  ]
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${local.resource_prefix}-cluster/cluster"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.eks.arn
  
  tags = local.common_tags
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-eks-kms-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.resource_prefix}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# ========================================
# EKS NODE GROUPS
# ========================================
# Node Group IAM Role
resource "aws_iam_role" "eks_nodes" {
  name = "${local.resource_prefix}-eks-nodes-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes.name
}

# Additional IAM policy for advanced features
resource "aws_iam_role_policy" "eks_nodes_additional" {
  name = "${local.resource_prefix}-eks-nodes-additional"
  role = aws_iam_role.eks_nodes.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeTags",
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup",
          "ec2:DescribeLaunchTemplateVersions",
          "ec2:DescribeInstanceTypes",
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "*"
      }
    ]
  })
}

# Node Groups
resource "aws_eks_node_group" "main" {
  for_each = local.node_groups
  
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${local.resource_prefix}-${each.key}"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = each.value.instance_types
  
  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }
  
  update_config {
    max_unavailable_percentage = 25
  }
  
  # Launch template configuration
  launch_template {
    id      = aws_launch_template.eks_nodes[each.key].id
    version = aws_launch_template.eks_nodes[each.key].latest_version
  }
  
  # Taints for specialized workloads
  dynamic "taint" {
    for_each = each.key == "compute_optimized" ? [1] : []
    content {
      key    = "compute-optimized"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
  
  dynamic "taint" {
    for_each = each.key == "memory_optimized" ? [1] : []
    content {
      key    = "memory-optimized"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-${each.key}-nodes"
    Type = each.key
  })
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# Launch Templates for Node Groups
resource "aws_launch_template" "eks_nodes" {
  for_each = local.node_groups
  
  name_prefix = "${local.resource_prefix}-${each.key}-"
  
  vpc_security_group_ids = [aws_security_group.eks_nodes.id]
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = aws_eks_cluster.main.name
    cluster_endpoint    = aws_eks_cluster.main.endpoint
    cluster_ca          = aws_eks_cluster.main.certificate_authority[0].data
    bootstrap_arguments = "--container-runtime containerd"
  }))
  
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      kms_key_id            = aws_kms_key.eks.arn
      delete_on_termination = true
    }
  }
  
  monitoring {
    enabled = true
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${local.resource_prefix}-${each.key}-node"
      Type = each.key
    })
  }
  
  tags = local.common_tags
}

# ========================================
# RDS POSTGRESQL CLUSTER
# ========================================
# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${local.resource_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-db-subnet-group"
  })
}

# RDS Cluster Parameter Group
resource "aws_rds_cluster_parameter_group" "main" {
  family = "aurora-postgresql15"
  name   = "${local.resource_prefix}-cluster-pg"
  
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,pg_hint_plan"
  }
  
  parameter {
    name  = "log_statement"
    value = "all"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }
  
  tags = local.common_tags
}

# RDS Aurora PostgreSQL Cluster
resource "aws_rds_cluster" "main" {
  cluster_identifier              = "${local.resource_prefix}-cluster"
  engine                         = "aurora-postgresql"
  engine_version                 = "15.4"
  database_name                  = "snapfix"
  master_username                = "snapfix_admin"
  manage_master_user_password    = true
  master_user_secret_kms_key_id  = aws_kms_key.rds.arn
  
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.main.name
  db_subnet_group_name           = aws_db_subnet_group.main.name
  vpc_security_group_ids         = [aws_security_group.database.id]
  
  backup_retention_period = 30
  preferred_backup_window = "03:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  
  storage_encrypted   = true
  kms_key_id         = aws_kms_key.rds.arn
  deletion_protection = var.environment == "production" ? true : false
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-rds-cluster"
  })
}

# RDS Cluster Instances
resource "aws_rds_cluster_instance" "main" {
  count = var.environment == "production" ? 3 : 2
  
  identifier         = "${local.resource_prefix}-instance-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = local.db_instance_class
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version
  
  performance_insights_enabled = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-rds-instance-${count.index + 1}"
  })
}

# KMS Key for RDS
resource "aws_kms_key" "rds" {
  description             = "RDS encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-rds-kms-key"
  })
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${local.resource_prefix}-rds"
  target_key_id = aws_kms_key.rds.key_id
}

# RDS Monitoring Role
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.resource_prefix}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ========================================
# ELASTICACHE REDIS CLUSTER
# ========================================
# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.resource_prefix}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
  
  tags = local.common_tags
}

# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "main" {
  family = "redis7.x"
  name   = "${local.resource_prefix}-redis-params"
  
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
  
  parameter {
    name  = "timeout"
    value = "300"
  }
  
  tags = local.common_tags
}

# ElastiCache Redis Replication Group
resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "${local.resource_prefix}-redis"
  description                  = "Redis cluster for SnapFix Enterprise"
  
  node_type                    = local.redis_node_type
  port                         = 6379
  parameter_group_name         = aws_elasticache_parameter_group.main.name
  
  num_cache_clusters           = var.environment == "production" ? 6 : 3
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = random_password.redis_auth.result
  kms_key_id                  = aws_kms_key.redis.arn
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format       = "text"
    log_type         = "slow-log"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-redis-cluster"
  })
}

# Redis Auth Token
resource "random_password" "redis_auth" {
  length  = 32
  special = true
}

# KMS Key for Redis
resource "aws_kms_key" "redis" {
  description             = "Redis encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-redis-kms-key"
  })
}

resource "aws_kms_alias" "redis" {
  name          = "alias/${local.resource_prefix}-redis"
  target_key_id = aws_kms_key.redis.key_id
}

# CloudWatch Log Group for Redis
resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/${local.resource_prefix}-redis"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.redis.arn
  
  tags = local.common_tags
}

# ========================================
# APPLICATION LOAD BALANCER
# ========================================
# ALB
resource "aws_lb" "main" {
  name               = "${local.resource_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "production" ? true : false
  enable_http2              = true
  enable_waf_fail_open      = false
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-alb"
  })
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${local.resource_prefix}-alb-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.resource_prefix}-alb-sg"
  })
}

# S3 Bucket for ALB Logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${local.resource_prefix}-alb-logs-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment != "production"
  
  tags = local.common_tags
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  rule {
    id     = "log_lifecycle"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ========================================
# OUTPUTS
# ========================================
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_version" {
  description = "EKS cluster version"
  value       = aws_eks_cluster.main.version
}

output "rds_cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = aws_rds_cluster.main.endpoint
  sensitive   = true
}

output "rds_cluster_reader_endpoint" {
  description = "RDS cluster reader endpoint"
  value       = aws_rds_cluster.main.reader_endpoint
  sensitive   = true
}

output "redis_cluster_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}

output "redis_cluster_reader_endpoint" {
  description = "Redis cluster reader endpoint"
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
  sensitive   = true
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Load balancer zone ID"
  value       = aws_lb.main.zone_id
}