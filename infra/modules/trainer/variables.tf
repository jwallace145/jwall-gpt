variable "project_name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  type        = string
  description = "Subnet for training workers."
}

variable "training_bucket_name" {
  type = string
}

variable "training_bucket_arn" {
  type = string
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type for GPU training."
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
  description = "Optional spot max price (empty = on-demand cap)."
  default     = ""
}

variable "assign_public_ip" {
  type        = bool
  description = "Assign a public IP (set false when using private subnets)."
  default     = true
}

variable "ami_id" {
  type        = string
  description = "Optional AMI override. Defaults to latest AWS Deep Learning GPU AMI."
  default     = ""
}

variable "tags" {
  type    = map(string)
  default = {}
}
