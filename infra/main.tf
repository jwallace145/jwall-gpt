terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_account.aws_region

  default_tags {
    tags = local.common_tags
  }
}

locals {
  common_tags = merge({
    Project    = var.project_name
    ManagedBy  = "terraform"
    Repository = "${var.github_details.github_org}/${var.github_details.github_repo}"
  }, var.aws_account.tags)
}

module "storage" {
  source = "./modules/storage"

  project_name              = var.project_name
  checkpoint_retention_days = var.aws_account.storage.checkpoint_retention_days
  log_retention_days        = var.aws_account.storage.log_retention_days
  dataset_noncurrent_days   = var.aws_account.storage.dataset_noncurrent_days
  state_noncurrent_days     = var.aws_account.storage.state_noncurrent_days
  abort_multipart_days      = var.aws_account.storage.abort_multipart_days
  tags                      = local.common_tags
}

module "vpc" {
  source = "./modules/vpc"

  project_name       = var.project_name
  vpc_cidr           = var.aws_account.network.vpc_cidr
  use_private_subnet = var.aws_account.network.use_private_subnet
  availability_zones = var.aws_account.network.availability_zones
  tags               = local.common_tags
}

module "trainer" {
  source = "./modules/trainer"

  project_name         = var.project_name
  vpc_id               = module.vpc.vpc_id
  subnet_id            = module.vpc.trainer_subnet_id
  training_bucket_name = module.storage.training_bucket_name
  training_bucket_arn  = module.storage.training_bucket_arn
  datasets_bucket_arn  = module.storage.datasets_bucket_arn
  instance_type        = var.aws_account.training_compute.instance_type
  root_volume_size_gb  = var.aws_account.training_compute.root_volume_size_gb
  use_spot_instances   = var.aws_account.training_compute.use_spot_instances
  spot_max_price       = var.aws_account.training_compute.spot_max_price
  assign_public_ip     = !var.aws_account.network.use_private_subnet
  ami_id               = var.aws_account.training_compute.ami_id
  tags                 = local.common_tags
}

module "github_oidc" {
  source = "./modules/github_oidc"

  project_name                = var.project_name
  github_org                  = var.github_details.github_org
  github_repo                 = var.github_details.github_repo
  terraform_state_bucket_arn  = module.storage.terraform_state_bucket_arn
  training_bucket_arn         = module.storage.training_bucket_arn
  trainer_instance_role_arn   = module.trainer.instance_role_arn
  create_github_oidc_provider = var.create_github_oidc_provider
  tags                        = local.common_tags
}

resource "aws_ssm_parameter" "training_bucket" {
  name  = "/${var.project_name}/training-bucket"
  type  = "String"
  value = module.storage.training_bucket_name
  tags  = local.common_tags
}

resource "aws_ssm_parameter" "datasets_bucket" {
  name  = "/${var.project_name}/datasets-bucket"
  type  = "String"
  value = module.storage.datasets_bucket_name
  tags  = local.common_tags
}

resource "aws_ssm_parameter" "github_actions_role_arn" {
  name  = "/${var.project_name}/github-actions-role-arn"
  type  = "String"
  value = module.github_oidc.role_arn
  tags  = local.common_tags
}

resource "aws_ssm_parameter" "aws_region" {
  name  = "/${var.project_name}/aws-region"
  type  = "String"
  value = var.aws_account.aws_region
  tags  = local.common_tags
}

resource "aws_ssm_parameter" "terraform_state_bucket" {
  name  = "/${var.project_name}/terraform-state-bucket"
  type  = "String"
  value = module.storage.terraform_state_bucket_name
  tags  = local.common_tags
}
