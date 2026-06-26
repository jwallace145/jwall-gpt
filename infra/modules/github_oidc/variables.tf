variable "project_name" {
  type = string
}

variable "github_org" {
  type        = string
  description = "GitHub organization or username that owns the repository."
}

variable "github_repo" {
  type        = string
  description = "GitHub repository name."
}

variable "terraform_state_bucket_arn" {
  type        = string
  description = "S3 bucket ARN used for Terraform remote state."
}

variable "training_bucket_arn" {
  type        = string
  description = "S3 bucket ARN for training data and checkpoints."
}

variable "trainer_instance_role_arn" {
  type        = string
  description = "IAM role ARN EC2 trainers assume (for iam:PassRole)."
}

variable "tags" {
  type    = map(string)
  default = {}
}
