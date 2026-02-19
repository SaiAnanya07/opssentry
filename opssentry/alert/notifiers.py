"""
Notification Integrations
Sends alerts via Email and Slack.
"""
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List
from datetime import datetime
import config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailNotifier:
    """Sends email notifications."""
    
    def __init__(self):
        """Initialize email notifier."""
        self.enabled = config.EMAIL_CONFIG['enabled']
        self.smtp_server = config.EMAIL_CONFIG['smtp_server']
        self.smtp_port = config.EMAIL_CONFIG['smtp_port']
        self.sender_email = config.EMAIL_CONFIG['sender_email']
        self.sender_password = config.EMAIL_CONFIG['sender_password']
        self.recipients = [r.strip() for r in config.EMAIL_CONFIG['recipient_emails'] if r.strip()]
    
    def send(self, subject: str, body: str, recipients: List[str] = None) -> bool:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Email body (HTML or plain text)
            recipients: List of recipient emails (uses config if None)
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info("Email notifications are disabled")
            return False
        
        if recipients is None:
            recipients = self.recipients
        
        if not recipients:
            logger.warning("No email recipients configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            
            # Add body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✓ Email sent to {len(recipients)} recipients")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class SlackNotifier:
    """Sends Slack notifications."""
    
    def __init__(self):
        """Initialize Slack notifier."""
        self.enabled = config.SLACK_CONFIG['enabled']
        self.webhook_url = config.SLACK_CONFIG['webhook_url']
    
    def send(self, message: str, blocks: List[Dict] = None) -> bool:
        """
        Send Slack notification.
        
        Args:
            message: Message text
            blocks: Optional Slack blocks for rich formatting
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info("Slack notifications are disabled")
            return False
        
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            payload = {'text': message}
            
            if blocks:
                payload['blocks'] = blocks
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("✓ Slack notification sent")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Send formatted alert to Slack.
        
        Args:
            alert_data: Alert information
        
        Returns:
            True if sent successfully
        """
        # Create rich Slack message with blocks
        risk_level = alert_data.get('risk_level', 'UNKNOWN')
        
        # Color coding
        color_map = {
            'CRITICAL': '#FF0000',
            'WARNING': '#FFA500',
            'INFO': '#0000FF',
            'LOW': '#00FF00'
        }
        color = color_map.get(risk_level, '#808080')
        
        # Create blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 OpsSentry Alert: {risk_level}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Pipeline:*\n{alert_data.get('run_name', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Prediction:*\n{alert_data.get('prediction_label', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Failure Probability:*\n{alert_data.get('failure_probability', 0):.1%}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Risk Level:*\n{risk_level}"
                    }
                ]
            }
        ]
        
        # Add recommendation if available
        if 'recommendation' in alert_data:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Recommendation:*\n{alert_data['recommendation']}"
                }
            })
        
        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Alert generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            ]
        })
        
        message = f"OpsSentry Alert: {risk_level} - {alert_data.get('run_name', 'Unknown Pipeline')}"
        
        return self.send(message, blocks)


class ConsoleNotifier:
    """Prints notifications to console/logs."""
    
    @staticmethod
    def send(message: str, alert_data: Dict[str, Any] = None) -> bool:
        """
        Print notification to console.
        
        Args:
            message: Message to print
            alert_data: Optional alert data for detailed logging
        
        Returns:
            Always True
        """
        logger.info("=" * 60)
        logger.info("ALERT NOTIFICATION")
        logger.info("=" * 60)
        logger.info(message)
        
        if alert_data:
            for key, value in alert_data.items():
                logger.info(f"{key:25s}: {value}")
        
        logger.info("=" * 60)
        
        return True


def main():
    """Test notifiers."""
    # Test data
    alert_data = {
        'run_name': 'CI Pipeline',
        'prediction_label': 'FAILURE',
        'failure_probability': 0.85,
        'risk_level': 'CRITICAL',
        'recommendation': 'Review recent code changes and increase test coverage.'
    }
    
    # Console notification (always works)
    console = ConsoleNotifier()
    console.send("Test alert", alert_data)
    
    # Email notification (if configured)
    email = EmailNotifier()
    if email.enabled:
        subject = f"OpsSentry Alert: {alert_data['risk_level']}"
        body = f"""
        <html>
        <body>
            <h2>Pipeline Failure Prediction</h2>
            <p><strong>Pipeline:</strong> {alert_data['run_name']}</p>
            <p><strong>Prediction:</strong> {alert_data['prediction_label']}</p>
            <p><strong>Failure Probability:</strong> {alert_data['failure_probability']:.1%}</p>
            <p><strong>Risk Level:</strong> {alert_data['risk_level']}</p>
            <p><strong>Recommendation:</strong> {alert_data['recommendation']}</p>
        </body>
        </html>
        """
        email.send(subject, body)
    
    # Slack notification (if configured)
    slack = SlackNotifier()
    if slack.enabled:
        slack.send_alert(alert_data)


if __name__ == "__main__":
    main()
