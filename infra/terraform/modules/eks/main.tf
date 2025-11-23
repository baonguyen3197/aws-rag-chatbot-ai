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
    
    subnet_ids     = var.private_subnet_ids
    
    scaling_config {
        desired_size = var.node_desired_size
        max_size     = var.node_max_size
        min_size     = var.node_min_size
    }
}