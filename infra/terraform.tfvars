# =========
# JWall GPT
# =========

project_name       = "jwall-gpt"

github_details = {
  github_org         = "jwallace145"
  github_repo        = "jwall-gpt"
}

aws_account = {
  aws_region         = "us-east-1"

  network = {
    vpc_cidr           = "10.0.0.0/16"
    use_private_subnet = false # set true for private trainers + NAT
  }

  training_compute = {
    instance_type        = "g4dn.xlarge"
    root_volume_size_gb  = 100
    use_spot_instances   = true
    spot_max_price       = "" # e.g. "0.25" to cap spot price
    # ami_id             = "" # optional AMI override
  }

  tags = {
    Owner = "james-wallace"
  }
}