output "cluster_name" {
    description = "Name of the EKS cluster"
    value       = aws_eks_cluster.this.name
}

output "node_group_name" {
    description = "Name of the EKS node group"
    value       = aws_eks_node_group.default.node_group_name
}

output "cluster_endpoint" {
    description = "Endpoint for the EKS cluster"
    value       = aws_eks_cluster.this.endpoint
}

output "cluster_certificate_authority_data" {
    description = "Certificate authority data for the EKS cluster"
    value       = aws_eks_cluster.this.certificate_authority[0].data
}