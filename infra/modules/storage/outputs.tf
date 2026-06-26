output "training_bucket_name" {
  value = aws_s3_bucket.training.id
}

output "training_bucket_arn" {
  value = aws_s3_bucket.training.arn
}

output "terraform_state_bucket_name" {
  value = aws_s3_bucket.terraform_state.id
}

output "terraform_state_bucket_arn" {
  value = aws_s3_bucket.terraform_state.arn
}
