terraform {
  backend "s3" {
    bucket       = "jwall-gpt-terraform-state"
    key          = "jwall-gpt/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
