variable "aws_region" {
  type        = string
  description = "AWS region for all resources."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Short project name used in resource names."
  default     = "jwall-gpt"
}

variable "github_org" {
  type        = string
  description = "GitHub org or username."
  default     = "jwallace145"
}

variable "github_repo" {
  type        = string
  description = "GitHub repository name."
  default     = "jwall-gpt"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "use_private_subnet" {
  type        = bool
  description = "Launch trainers in a private subnet with NAT egress."
  default     = false
}

variable "availability_zones" {
  type        = list(string)
  description = "Optional AZ override (defaults to first two available)."
  default     = []
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type for training workers."
  default     = "g4dn.xlarge"
}

variable "root_volume_size_gb" {
  type    = number
  default = 100
}

variable "use_spot_instances" {
  type    = bool
  default = true
}

variable "spot_max_price" {
  type        = string
  description = "Optional spot max price. Empty uses the on-demand price cap."
  default     = ""
}

variable "ami_id" {
  type        = string
  description = "Optional AMI override for trainers."
  default     = ""
}

variable "create_github_oidc_provider" {
  type        = bool
  description = "Create the GitHub OIDC provider. Leave false if it already exists in your AWS account."
  default     = false
}

variable "tags" {
  type        = map(string)
  description = "Additional resource tags."
  default     = {}
}
