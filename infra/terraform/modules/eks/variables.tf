variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "vpc_id" {
  type        = string
  description = "VPC id to place cluster resources into"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for node groups"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnet IDs for node groups (optional, for nodes with public IPs)"
  default     = []
}

variable "use_public_subnets" {
  type        = bool
  description = "Whether to deploy nodes in public subnets instead of private"
  default     = false
}

variable "eks_cluster_role" {
  type        = string
  description = "ARN of the IAM role to use for the EKS control plane"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.34"
}

variable "node_role_arn" {
  description = "ARN of the IAM role to associate with the EKS worker nodes"
  type        = string
}

variable "node_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 3
}

variable "node_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  type        = string
  default     = "m5.xlarge"
}

variable "node_volume_size" {
  description = "Disk size in GiB for EKS worker nodes"
  type        = number
  default     = 20
}

variable "enable_ebs_addon" {
  description = "Whether to create the EBS CSI addon and related IAM role/policy"
  type        = bool
  default     = true
}

variable "ebs_service_account_namespace" {
  description = "Namespace of the EBS CSI driver service account"
  type        = string
  default     = "kube-system"
}

variable "ebs_service_account_name" {
  description = "Service account name used by the EBS CSI controller"
  type        = string
  default     = "ebs-csi-controller-sa"
}

variable "create_iam_oidc_provider" {
  description = "Whether to create an IAM OIDC provider for the EKS cluster (required for IRSA). If false, assume provider already exists."
  type        = bool
  default     = true
}

variable "ebs_role_name" {
  description = "IAM role name to create for EBS CSI (IRSA)"
  type        = string
  default     = "nhqb-terraform-EBS"
}

variable "enable_addon_vpc_cni" {
  description = "Enable the Amazon VPC CNI managed addon"
  type        = bool
  default     = true
}

variable "enable_addon_coredns" {
  description = "Enable the CoreDNS managed addon"
  type        = bool
  default     = true
}

variable "enable_addon_kube_proxy" {
  description = "Enable the kube-proxy managed addon"
  type        = bool
  default     = true
}

variable "enable_addon_pod_identity" {
  description = "Enable the Amazon EKS Pod Identity Agent managed addon"
  type        = bool
  default     = false
}

variable "ebs_service_account_role_arn" {
  description = "Optional: existing IAM role ARN to attach to the EBS CSI driver's service account via the addon"
  type        = string
  default     = ""
}

variable "pod_identity_service_account_role_arn" {
  description = "Optional: existing IAM role ARN to attach to the Pod Identity Agent service account via the addon"
  type        = string
  default     = ""
}