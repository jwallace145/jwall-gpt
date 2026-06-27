# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is maintained by [python-semantic-release](https://python-semantic-release.readthedocs.io/).

<!-- version list -->

## v0.4.0 (2026-06-27)

### Features

- Apply Terraform automatically on infra changes to main
  ([#3](https://github.com/jwallace145/jwall-gpt/pull/3),
  [`be1cc26`](https://github.com/jwallace145/jwall-gpt/commit/be1cc2674da591d13f26b2d281c1f5f4ca740a99))

### Refactoring

- Default trainer instances to on-demand ([#2](https://github.com/jwallace145/jwall-gpt/pull/2),
  [`8d79793`](https://github.com/jwallace145/jwall-gpt/commit/8d79793e6cbcafdbf54507c163a6cf4f7f509ce3))


## v0.3.0 (2026-06-27)

### Bug Fixes

- Use Terraform 1.11 for S3 native state locking
  ([`b6b106c`](https://github.com/jwallace145/jwall-gpt/commit/b6b106cf56a57228d71f60b0642496f781747ce8))

### Chores

- Add pull request template
  ([`a00b0ce`](https://github.com/jwallace145/jwall-gpt/commit/a00b0ce1ab17330fcec2478563c7a8c2a32d2c9f))

- Pin Terraform version at repo root
  ([`3531bc7`](https://github.com/jwallace145/jwall-gpt/commit/3531bc7a133578f7db9c852a3a7e29a21623c024))

- Track Terraform backend.tf and terraform.tfvars in git
  ([`ae4faeb`](https://github.com/jwallace145/jwall-gpt/commit/ae4faebc4789c5c55dc79dc1f60228ac37ace4b7))

### Code Style

- Run terraform fmt on infra files
  ([`55ce7fa`](https://github.com/jwallace145/jwall-gpt/commit/55ce7fa44e647bfe2d9bcad7e4a05f33dad0b03f))

### Features

- Require training environment approval before AWS launches
  ([`7a5ef14`](https://github.com/jwallace145/jwall-gpt/commit/7a5ef145d6bdb9c0a8ee3a78daf2de257455bcc7))

### Refactoring

- Nest Terraform variables to match tfvars interface
  ([`fae5113`](https://github.com/jwallace145/jwall-gpt/commit/fae5113b5ffc5517adf0da711cf30d26fa58369c))

- Use fixed name for trainer launch template
  ([`d1c4179`](https://github.com/jwallace145/jwall-gpt/commit/d1c4179911b4c86bf50ca197106779cf62f9e7df))


## v0.2.0 (2026-06-26)

### Features

- Add AWS training pipeline and enforce single-line commits
  ([`0e805d0`](https://github.com/jwallace145/jwall-gpt/commit/0e805d0fbeb33dfc8ecb6f927e9b280935a77ce1))


## v0.1.1 (2026-06-26)

### Bug Fixes

- Configure PSR changelog insertion flag and backfill v0.1.0
  ([`9cbc59c`](https://github.com/jwallace145/jwall-gpt/commit/9cbc59cdb96c7ec03ba223e9de101d239b37a877))


## v0.1.0 (2026-06-26)

_This release is published under the MIT License._

- Initial Release
