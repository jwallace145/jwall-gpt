variable "project_name" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

variable "checkpoint_retention_days" {
  type        = number
  description = "Days to keep model checkpoints in the training bucket before expiry."
  default     = 90
}

variable "log_retention_days" {
  type        = number
  description = "Days to keep training logs in the training bucket before expiry."
  default     = 30
}

variable "dataset_noncurrent_days" {
  type        = number
  description = "Days to keep superseded (noncurrent) dataset object versions before expiry."
  default     = 30
}

variable "state_noncurrent_days" {
  type        = number
  description = "Days to keep superseded (noncurrent) Terraform state versions before expiry."
  default     = 90
}

variable "abort_multipart_days" {
  type        = number
  description = "Days after which incomplete multipart uploads are aborted to avoid orphaned storage cost."
  default     = 7
}
