variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "azs" {
  description = "List of availability zones"
  type        = list(string)
}

variable "project_name" {
  description = "Small terraform project"
  type        = string
  default     = "nhqb-terraform"
}

variable "public_subnets" {
  type        = list(string)
  description = "CIDRs for public subnets"
  default     = ["10.0.0.0/26", "10.0.0.64/26"]
}

variable "private_subnets" {
  type        = list(string)
  description = "CIDRs for private subnets"
  default     = ["10.0.0.128/26", "10.0.0.192/26"]
}