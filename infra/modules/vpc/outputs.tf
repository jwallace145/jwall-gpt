output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "trainer_subnet_id" {
  description = "Subnet where training workers are launched."
  value       = var.use_private_subnet ? aws_subnet.private[0].id : aws_subnet.public[0].id
}
