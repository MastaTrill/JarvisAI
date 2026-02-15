# Jarvis AI - Terraform Infrastructure
# Main configuration for multi-cloud deployment

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "jarvis-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "jarvis-terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "jarvis-ai-cluster"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 3
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 20
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 5
}

# Provider configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "JarvisAI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "jarvis-ai-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = false
  one_nat_gateway_per_az = true
  enable_dns_hostnames   = true
  enable_dns_support     = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  enable_irsa = true

  eks_managed_node_groups = {
    jarvis-nodes = {
      name = "jarvis-node-group"

      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      labels = {
        role = "jarvis-worker"
      }

      update_config = {
        max_unavailable_percentage = 33
      }
    }
    
    jarvis-gpu-nodes = {
      name = "jarvis-gpu-node-group"

      instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]
      capacity_type  = "ON_DEMAND"
      ami_type       = "AL2_x86_64_GPU"

      min_size     = 0
      max_size     = 5
      desired_size = 1

      labels = {
        role = "jarvis-gpu"
        "nvidia.com/gpu" = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
}

# RDS PostgreSQL
resource "aws_db_subnet_group" "jarvis" {
  name       = "jarvis-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name        = "jarvis-rds-sg"
  description = "Security group for Jarvis RDS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "jarvis" {
  identifier = "jarvis-db"

  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.r6g.large"
  allocated_storage    = 100
  max_allocated_storage = 500
  storage_type         = "gp3"
  storage_encrypted    = true

  db_name  = "jarvis"
  username = "jarvis_admin"
  password = var.db_password  # Pass via TF_VAR_db_password

  multi_az               = true
  db_subnet_group_name   = aws_db_subnet_group.jarvis.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  performance_insights_enabled = true
  monitoring_interval         = 60

  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "jarvis-db-final-snapshot"
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "jarvis" {
  name       = "jarvis-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name        = "jarvis-redis-sg"
  description = "Security group for Jarvis Redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_elasticache_replication_group" "jarvis" {
  replication_group_id = "jarvis-redis"
  description          = "Jarvis AI Redis cluster"

  node_type            = "cache.r6g.large"
  num_cache_clusters   = 3
  port                 = 6379

  subnet_group_name    = aws_elasticache_subnet_group.jarvis.name
  security_group_ids   = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token  # Pass via TF_VAR_redis_auth_token

  automatic_failover_enabled = true
  multi_az_enabled          = true

  snapshot_retention_limit = 7
  snapshot_window         = "02:00-03:00"

  engine_version = "7.0"
  parameter_group_name = "default.redis7"
}

# S3 Bucket for ML Models and Data
resource "aws_s3_bucket" "jarvis_data" {
  bucket = "jarvis-ai-data-${var.environment}"
}

resource "aws_s3_bucket_versioning" "jarvis_data" {
  bucket = aws_s3_bucket.jarvis_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "jarvis_data" {
  bucket = aws_s3_bucket.jarvis_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "jarvis_data" {
  bucket = aws_s3_bucket.jarvis_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.jarvis.endpoint
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.jarvis.primary_endpoint_address
}

output "s3_bucket" {
  description = "S3 bucket for data"
  value       = aws_s3_bucket.jarvis_data.id
}

# Sensitive variables (passed via environment)
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
}
