variable "project_name" {
  type        = string
  description = "Project name used for resource naming."
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC."
  default     = "10.0.0.0/16"
}

variable "use_private_subnet" {
  type        = bool
  description = "When true, trainers run in a private subnet (NAT required for egress)."
  default     = false
}

variable "availability_zones" {
  type        = list(string)
  description = "AZs to spread subnets across."
  default     = []
}

variable "tags" {
  type        = map(string)
  description = "Tags applied to all VPC resources."
  default     = {}
}
