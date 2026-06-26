output "launch_template_id" {
  value = aws_launch_template.trainer.id
}

output "launch_template_latest_version" {
  value = aws_launch_template.trainer.latest_version
}

output "security_group_id" {
  value = aws_security_group.trainer.id
}

output "instance_role_arn" {
  value = aws_iam_role.trainer.arn
}

output "instance_profile_name" {
  value = aws_iam_instance_profile.trainer.name
}
