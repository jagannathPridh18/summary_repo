output "public_ip" {
  description = "Elastic IP of the instance. Create a DNS A-record: <domain> -> this IP."
  value       = aws_eip.this.public_ip
}

output "instance_id" {
  description = "EC2 instance id."
  value       = aws_instance.this.id
}

output "ami_id" {
  description = "Deep Learning GPU AMI the instance booted from."
  value       = data.aws_ami.dlami_gpu.id
}

output "domain" {
  description = "Hostname Caddy serves."
  value       = var.domain
}

output "ssh_command" {
  description = "SSH into the instance."
  value       = "ssh -i ${var.project_name}-key.pem ubuntu@${aws_eip.this.public_ip}"
}

output "private_key_path" {
  description = "Local path to the generated SSH private key (also the CI/CD GitHub secret EC2_SSH_KEY)."
  value       = local_sensitive_file.private_key.filename
}

output "health_url" {
  description = "Health endpoint once DNS + HTTPS are live."
  value       = "https://${var.domain}/health"
}

output "health_url_ip" {
  description = "Health endpoint reachable by IP over HTTP before DNS propagates."
  value       = "http://${aws_eip.this.public_ip}/health"
}

output "docs_url" {
  description = "Interactive Swagger docs once live."
  value       = "https://${var.domain}/docs"
}
