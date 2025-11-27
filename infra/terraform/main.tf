terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "./modules/vpc"

  vpc_cidr = var.vpc_cidr
  azs      = var.azs
}

locals {
  cluster_name = var.cluster_name != null ? var.cluster_name : "${var.project_name}-k8s-cluster"
}

module "eks" {
  source = "./modules/eks"

  cluster_name       = local.cluster_name
  kubernetes_version = var.kubernetes_version
  eks_cluster_role   = aws_iam_role.eks_cluster.arn
  node_role_arn      = aws_iam_role.eks_node.arn
  node_desired_size  = var.node_desired_size
  node_min_size      = var.node_min_size
  node_max_size      = var.node_max_size
  node_instance_type = var.node_instance_type

  private_subnet_ids = module.vpc.private_subnet_ids
  vpc_id             = module.vpc.vpc_id
  # enable the managed EKS addons; attach existing ttb-EBS role to EBS CSI and Pod Identity Agent
  ebs_role_name             = "ttb-EBS"
  enable_ebs_addon          = var.enable_ebs_addon
  enable_addon_vpc_cni      = true
  enable_addon_coredns      = true
  enable_addon_kube_proxy   = true
  enable_addon_pod_identity = true
  create_iam_oidc_provider  = var.create_iam_oidc_provider
  # pass existing role ARNs so the addons attach to the ttb-EBS role
  ebs_service_account_role_arn          = "arn:aws:iam::906034468113:role/ttb-EBS"
  pod_identity_service_account_role_arn = "arn:aws:iam::906034468113:role/ttb-EBS"
}