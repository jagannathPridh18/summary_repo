# ─────────────────────────────────────────────────────────────────────
# CloudWatch log groups for the logs the CW agent ships (see the
# logs.collect_list in user_data.sh.tpl). The agent auto-creates these on
# first boot with NO retention (never expire); declaring them here codifies
# the retention policy so it survives and is managed by Terraform.
#
# NOTE: if the agent creates them before Terraform does, import them:
#   terraform import aws_cloudwatch_log_group.bootstrap /call-chat-summarizer/bootstrap
#   terraform import aws_cloudwatch_log_group.app       /call-chat-summarizer/app
# ─────────────────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "bootstrap" {
  name              = "/${var.project_name}/bootstrap"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/${var.project_name}/app"
  retention_in_days = var.log_retention_days
}
