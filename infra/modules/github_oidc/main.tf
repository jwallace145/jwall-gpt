variable "create_github_oidc_provider" {
  type        = bool
  description = "Create the account-wide GitHub OIDC provider. Set false if https://token.actions.githubusercontent.com already exists (default for most accounts)."
  default     = false
}

locals {
  github_oidc_url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = local.github_oidc_url
}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 1 : 0

  url             = local.github_oidc_url
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

locals {
  github_oidc_provider_arn = var.create_github_oidc_provider ? aws_iam_openid_connect_provider.github[0].arn : data.aws_iam_openid_connect_provider.github[0].arn
}

data "aws_iam_policy_document" "github_oidc_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_org}/${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.project_name}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_oidc_assume.json

  tags = var.tags
}

data "aws_iam_policy_document" "github_actions" {
  statement {
    sid    = "TerraformState"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      var.terraform_state_bucket_arn,
      "${var.terraform_state_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "TerraformRead"
    effect = "Allow"
    actions = [
      "ec2:Describe*",
      "iam:Get*",
      "iam:List*",
      "s3:Get*",
      "s3:List*",
      "ssm:GetParameter",
      "ssm:GetParameters",
      "ssm:DescribeParameters",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "TerraformWrite"
    effect = "Allow"
    actions = [
      "ec2:*",
      "iam:*",
      "s3:*",
      "ssm:*",
      "logs:*",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "LaunchTrainers"
    effect = "Allow"
    actions = [
      "ec2:RunInstances",
      "ec2:TerminateInstances",
      "ec2:CreateTags",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassTrainerRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [var.trainer_instance_role_arn]
  }

  statement {
    sid    = "TrainingBucketAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      var.training_bucket_arn,
      "${var.training_bucket_arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "github_actions" {
  name   = "${var.project_name}-github-actions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions.json
}
