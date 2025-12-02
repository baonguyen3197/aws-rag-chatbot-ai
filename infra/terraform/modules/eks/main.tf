resource "aws_eks_cluster" "this" {
    name     = var.cluster_name
    role_arn = var.eks_cluster_role

    vpc_config {
        subnet_ids = var.private_subnet_ids
    }

    version = var.kubernetes_version
}

resource "aws_eks_node_group" "default" {
    cluster_name   = aws_eks_cluster.this.name
    node_role_arn  = var.node_role_arn
    node_group_name = "${var.cluster_name}-node-group"
    
    subnet_ids     = var.use_public_subnets ? var.public_subnet_ids : var.private_subnet_ids
    instance_types = [var.node_instance_type]
    disk_size      = var.node_volume_size
    
    scaling_config {
        desired_size = var.node_desired_size
        max_size     = var.node_max_size
        min_size     = var.node_min_size
    }
}

    data "aws_caller_identity" "current" {}

locals {
    oidc_provider = replace(aws_eks_cluster.this.identity[0].oidc[0].issuer, "https://", "")
    account_id    = data.aws_caller_identity.current.account_id
}

    # Fetch the OIDC server certificate so we can compute the thumbprint for the IAM provider
    data "tls_certificate" "oidc" {
        url = aws_eks_cluster.this.identity[0].oidc[0].issuer
    }

    # Optionally create the IAM OIDC provider for the cluster so IRSA works
    resource "aws_iam_openid_connect_provider" "oidc" {
        count           = var.create_iam_oidc_provider ? 1 : 0
        url             = aws_eks_cluster.this.identity[0].oidc[0].issuer
        client_id_list  = ["sts.amazonaws.com"]
        thumbprint_list = [data.tls_certificate.oidc.certificates[0].sha1_fingerprint]
    }

# IAM policy for EBS CSI driver
resource "aws_iam_policy" "ebs_csi" {
    count       = var.enable_ebs_addon ? 1 : 0
    name        = "${var.cluster_name}-ebs-csi-policy"
    description = "Policy for EBS CSI driver to manage EBS volumes"

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Effect = "Allow"
                Action = [
                    "ec2:CreateVolume",
                    "ec2:DeleteVolume",
                    "ec2:AttachVolume",
                    "ec2:DetachVolume",
                    "ec2:ModifyVolume",
                    "ec2:DescribeVolumes",
                    "ec2:DescribeVolumesModifications",
                    "ec2:DescribeInstances",
                    "ec2:DescribeAvailabilityZones",
                    "ec2:CreateTags",
                    "ec2:DeleteTags",
                    "ec2:DescribeTags"
                ]
                Resource = "*"
            },
            {
                Effect = "Allow"
                Action = [
                    "kms:CreateGrant",
                    "kms:DescribeKey",
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*"
                ]
                Resource = "*"
            }
        ]
    })
}

# IAM role for EBS CSI driver (IRSA)
resource "aws_iam_role" "nhqb_terraform_ebs" {
        count = var.enable_ebs_addon ? 1 : 0
        name  = var.ebs_role_name

        assume_role_policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::${local.account_id}:oidc-provider/${local.oidc_provider}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringLike": {
                    "${local.oidc_provider}:sub": "system:serviceaccount:${var.ebs_service_account_namespace}:${var.ebs_service_account_name}"
                }
            }
        }
    ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "ebs_attach" {
        count      = var.enable_ebs_addon ? 1 : 0
        role       = aws_iam_role.nhqb_terraform_ebs[0].name
        policy_arn = aws_iam_policy.ebs_csi[0].arn
}

# Create the managed EKS addon for the AWS EBS CSI driver and assign the IRSA role
resource "aws_eks_addon" "ebs" {
  count        = var.enable_ebs_addon ? 1 : 0

  addon_name   = "aws-ebs-csi-driver"
  cluster_name = aws_eks_cluster.this.name

  # Role used by the EBS CSI driver
  service_account_role_arn = (
    length(var.ebs_service_account_role_arn) > 0
    ? var.ebs_service_account_role_arn
    : aws_iam_role.nhqb_terraform_ebs[0].arn
  )

  # Required with recent AWS provider versions
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [
        aws_eks_node_group.default,          # your node group resource
        aws_iam_role.nhqb_terraform_ebs[0],  # role for CSI addon (indexed because of count)
        aws_iam_openid_connect_provider.oidc[0]
  ]
}

# Amazon VPC CNI managed addon
resource "aws_eks_addon" "vpc_cni" {
    count        = var.enable_addon_vpc_cni ? 1 : 0
    addon_name   = "vpc-cni"
    cluster_name = aws_eks_cluster.this.name
    resolve_conflicts_on_create = "OVERWRITE"
    resolve_conflicts_on_update = "OVERWRITE"
}

# CoreDNS managed addon
resource "aws_eks_addon" "coredns" {
    count        = var.enable_addon_coredns ? 1 : 0
    addon_name   = "coredns"
    cluster_name = aws_eks_cluster.this.name
    resolve_conflicts_on_create = "OVERWRITE"
    resolve_conflicts_on_update = "OVERWRITE"
    
    depends_on = [
        aws_eks_node_group.default,
        aws_eks_addon.vpc_cni,
        aws_eks_addon.ebs
    ]
}

# kube-proxy managed addon
resource "aws_eks_addon" "kube_proxy" {
    count        = var.enable_addon_kube_proxy ? 1 : 0
    addon_name   = "kube-proxy"
    cluster_name = aws_eks_cluster.this.name
    resolve_conflicts_on_create = "OVERWRITE"
    resolve_conflicts_on_update = "OVERWRITE"
}

# Pod Identity Agent managed addon (attachable service account role optional)
resource "aws_eks_addon" "pod_identity" {
    count        = var.enable_addon_pod_identity ? 1 : 0
    addon_name   = "eks-pod-identity-agent"
    cluster_name = aws_eks_cluster.this.name
    service_account_role_arn = length(var.pod_identity_service_account_role_arn) > 0 ? var.pod_identity_service_account_role_arn : null
    resolve_conflicts_on_create = "OVERWRITE"
    resolve_conflicts_on_update = "OVERWRITE"
}