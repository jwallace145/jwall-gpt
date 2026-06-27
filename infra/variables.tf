variable "project_name" {
  type        = string
  description = "Short project name used in resource names."
}

variable "github_details" {
  type = object({
    github_org  = string
    github_repo = string
  })
  description = "GitHub repository identity for OIDC trust and tagging."
}

variable "aws_account" {
  type = object({
    aws_region = string
    network = object({
      vpc_cidr             = string
      use_private_subnet   = bool
      availability_zones   = optional(list(string), [])
    })
    training_compute = object({
      instance_type       = string
      root_volume_size_gb = number
      use_spot_instances  = bool
      spot_max_price      = optional(string, "")
      ami_id              = optional(string, "")
    })
    tags = optional(map(string), {})
  })
  description = "AWS region, networking, trainer compute, and resource tags."
}

variable "create_github_oidc_provider" {
  type        = bool
  description = "Create the GitHub OIDC provider. Leave false if it already exists in your AWS account."
  default     = false
}
