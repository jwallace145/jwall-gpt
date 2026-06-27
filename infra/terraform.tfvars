# =========
# JWall GPT
# =========

project_name = "jwall-gpt"

github_details = {
  github_org  = "jwallace145"
  github_repo = "jwall-gpt"
}

aws_account = {
  aws_region = "us-east-1"

  network = {
    vpc_cidr           = "10.0.0.0/16"
    use_private_subnet = false # set true for private trainers + NAT
  }

  training_compute = {
    instance_type       = "g4dn.xlarge"
    root_volume_size_gb = 100
    # ami_id             = "" # optional AMI override
  }

  # S3 lifecycle retention (days). Outputs expire; durable inputs/state keep
  # only their current version and trim old version history.
  storage = {
    checkpoint_retention_days = 90 # model checkpoints in the training bucket
    log_retention_days        = 30 # training logs in the training bucket
    dataset_noncurrent_days   = 30 # superseded dataset versions
    state_noncurrent_days     = 90 # superseded Terraform state versions
    abort_multipart_days      = 7  # abort incomplete multipart uploads
  }

  tags = {
    Owner = "james-wallace"
  }
}