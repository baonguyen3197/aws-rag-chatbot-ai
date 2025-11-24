aws-eks-terraform/
├── main.tf
├── README.md
├── terraform.tfvars
├── variables.tf
├── outputs.tf
└── modules/
    ├── vpc/
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── eks/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf

=====================================
## AWS EKS Terraform Module
=====================================

## Insallation
```bash
sudo apt-get update && sudo apt-get install -y terraform
```

## Initialization
```bash
terraform init
```

## Plan Terraform Configuration
```bash
terraform fmt
terraform validate
terraform plan --out=tfplan
```

## Apply Terraform Configuration
```bash
terraform apply "tfplan"
```

## Destroy Terraform-managed Infrastructure
```bash
terraform destroy
```