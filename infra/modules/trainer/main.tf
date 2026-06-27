data "aws_region" "current" {}

data "aws_ami" "gpu" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.gpu[0].id
}

resource "aws_security_group" "trainer" {
  name        = "${var.project_name}-trainer"
  description = "Security group for jwall-gpt training workers"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-trainer-sg"
  })
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "trainer" {
  name               = "${var.project_name}-trainer"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "trainer" {
  statement {
    sid    = "TrainingBucket"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject",
    ]
    resources = [
      var.training_bucket_arn,
      "${var.training_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "DatasetsBucketReadOnly"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      var.datasets_bucket_arn,
      "${var.datasets_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:${data.aws_region.current.name}:*:*"]
  }

}

resource "aws_iam_role_policy" "trainer" {
  name   = "${var.project_name}-trainer"
  role   = aws_iam_role.trainer.id
  policy = data.aws_iam_policy_document.trainer.json
}

resource "aws_iam_instance_profile" "trainer" {
  name = "${var.project_name}-trainer"
  role = aws_iam_role.trainer.name
}

resource "aws_launch_template" "trainer" {
  name          = "${var.project_name}-trainer"
  image_id      = local.ami_id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.trainer.name
  }

  vpc_security_group_ids = [aws_security_group.trainer.id]

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = var.root_volume_size_gb
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot_instances ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price = var.spot_max_price != "" ? var.spot_max_price : null
      }
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name    = "${var.project_name}-trainer"
      Project = var.project_name
      Role    = "trainer"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(var.tags, {
      Name    = "${var.project_name}-trainer-volume"
      Project = var.project_name
    })
  }

}

resource "aws_ssm_parameter" "launch_template_id" {
  name  = "/${var.project_name}/launch-template-id"
  type  = "String"
  value = aws_launch_template.trainer.id
  tags  = var.tags
}

resource "aws_ssm_parameter" "trainer_subnet_id" {
  name  = "/${var.project_name}/trainer-subnet-id"
  type  = "String"
  value = var.subnet_id
  tags  = var.tags
}

resource "aws_ssm_parameter" "trainer_instance_profile" {
  name  = "/${var.project_name}/trainer-instance-profile"
  type  = "String"
  value = aws_iam_instance_profile.trainer.name
  tags  = var.tags
}

resource "aws_ssm_parameter" "assign_public_ip" {
  name  = "/${var.project_name}/trainer-assign-public-ip"
  type  = "String"
  value = tostring(var.assign_public_ip)
  tags  = var.tags
}
