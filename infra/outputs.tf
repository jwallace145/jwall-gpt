output "training_bucket_name" {
  value = module.storage.training_bucket_name
}

output "terraform_state_bucket_name" {
  value = module.storage.terraform_state_bucket_name
}

output "github_actions_role_arn" {
  value = module.github_oidc.role_arn
}

output "launch_template_id" {
  value = module.trainer.launch_template_id
}

output "trainer_subnet_id" {
  value = module.vpc.trainer_subnet_id
}

output "trainer_security_group_id" {
  value = module.trainer.security_group_id
}

output "vpc_id" {
  value = module.vpc.vpc_id
}
