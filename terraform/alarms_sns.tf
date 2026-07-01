# ─────────────────────────────────────────────────────────────────────
# SNS topic + email subscription for CloudWatch alarm notifications.
# The 3 alarms (status-check, cpu-high, gpu-mem-high) publish ALARM and OK
# transitions here (see alarm_actions / ok_actions in main.tf).
#
# NOTE: an email subscription is created in "pending confirmation" state —
# AWS emails var.alarm_email a confirmation link that MUST be clicked once
# before any notification is delivered. Terraform cannot auto-confirm it.
# ─────────────────────────────────────────────────────────────────────
resource "aws_sns_topic" "alarms" {
  name = "${var.project_name}-alarms"
}

resource "aws_sns_topic_subscription" "alarms_email" {
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

output "alarm_sns_topic_arn" {
  description = "SNS topic the CloudWatch alarms publish to."
  value       = aws_sns_topic.alarms.arn
}

output "alarm_email_pending_confirmation" {
  description = "Email that must confirm the SNS subscription before alerts are delivered."
  value       = var.alarm_email
}
