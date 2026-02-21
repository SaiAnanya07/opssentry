"""
Alert Manager
Manages alert generation and routing to notification channels.
"""
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import config
from utils.logger import setup_logger
from alert.notifiers import EmailNotifier, SlackNotifier, ConsoleNotifier

logger = setup_logger(__name__)


class AlertManager:
    """Manages alert generation and distribution."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.email_notifier = EmailNotifier()
        self.slack_notifier = SlackNotifier()
        self.console_notifier = ConsoleNotifier()
        self.alert_history = []
    
    def should_alert(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if an alert should be sent based on prediction.
        
        Args:
            prediction: Prediction result from Predictor
        
        Returns:
            True if alert should be sent
        """
        return prediction.get('should_alert', False)
    
    def create_alert(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create alert message from prediction.
        
        Args:
            prediction: Prediction result
        
        Returns:
            Alert data dict
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'run_id': prediction.get('run_id'),
            'run_name': prediction.get('run_name', 'Unknown Pipeline'),
            'prediction_label': prediction.get('prediction_label'),
            'failure_probability': prediction.get('failure_probability'),
            'success_probability': prediction.get('success_probability'),
            'risk_level': prediction.get('risk_level'),
            'recommendation': prediction.get('recommendation', 'Monitor pipeline execution.'),
        }
        
        return alert
    
    def send_alert(self, alert: Dict[str, Any], channels: List[str] = None) -> Dict[str, bool]:
        """
        Send alert to specified channels.
        
        Args:
            alert: Alert data
            channels: List of channels ('email', 'slack', 'console')
                     If None, sends to all enabled channels
        
        Returns:
            Dict mapping channel names to success status
        """
        if channels is None:
            channels = ['console']  # Always send to console
            if self.email_notifier.enabled:
                channels.append('email')
            if self.slack_notifier.enabled:
                channels.append('slack')
        
        results = {}
        
        # Console notification
        if 'console' in channels:
            message = (f"Pipeline '{alert['run_name']}' predicted to {alert['prediction_label']} "
                      f"with {alert['failure_probability']:.1%} probability")
            results['console'] = self.console_notifier.send(message, alert)
        
        # Email notification
        if 'email' in channels and self.email_notifier.enabled:
            subject = f"OpsSentry Alert: {alert['risk_level']} - {alert['run_name']}"
            body = self._create_email_body(alert)
            results['email'] = self.email_notifier.send(subject, body)
        
        # Slack notification
        if 'slack' in channels and self.slack_notifier.enabled:
            results['slack'] = self.slack_notifier.send_alert(alert)
        
        # Store in history
        alert['sent_to'] = [ch for ch, success in results.items() if success]
        self.alert_history.append(alert)
        
        logger.info(f"Alert sent to {len(alert['sent_to'])} channels: {alert['sent_to']}")
        
        return results
    
    def _create_email_body(self, alert: Dict[str, Any]) -> str:
        """
        Create HTML email body from alert data.
        
        Args:
            alert: Alert data
        
        Returns:
            HTML string
        """
        risk_color = {
            'CRITICAL': '#FF0000',
            'WARNING': '#FFA500',
            'INFO': '#0000FF',
            'LOW': '#00FF00'
        }.get(alert['risk_level'], '#808080')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {risk_color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metric {{ margin: 10px 0; }}
                .metric-label {{ font-weight: bold; }}
                .recommendation {{ background-color: #f0f0f0; padding: 15px; margin-top: 20px; border-left: 4px solid {risk_color}; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 OpsSentry Alert: {alert['risk_level']}</h1>
            </div>
            <div class="content">
                <div class="metric">
                    <span class="metric-label">Pipeline:</span> {alert['run_name']}
                </div>
                <div class="metric">
                    <span class="metric-label">Prediction:</span> {alert['prediction_label']}
                </div>
                <div class="metric">
                    <span class="metric-label">Failure Probability:</span> {alert['failure_probability']:.1%}
                </div>
                <div class="metric">
                    <span class="metric-label">Success Probability:</span> {alert['success_probability']:.1%}
                </div>
                <div class="metric">
                    <span class="metric-label">Risk Level:</span> {alert['risk_level']}
                </div>
                <div class="metric">
                    <span class="metric-label">Timestamp:</span> {alert['timestamp']}
                </div>
                
                <div class="recommendation">
                    <h3>💡 Recommendation</h3>
                    <p>{alert['recommendation']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def process_prediction(self, prediction: Dict[str, Any], 
                          channels: List[str] = None) -> bool:
        """
        Process prediction and send alert if needed.
        
        Args:
            prediction: Prediction result from Predictor
            channels: Channels to send alert to
        
        Returns:
            True if alert was sent
        """
        if self.should_alert(prediction):
            alert = self.create_alert(prediction)
            self.send_alert(alert, channels)
            return True
        else:
            logger.info(f"No alert needed for {prediction.get('run_name')} "
                       f"(probability: {prediction.get('failure_probability', 0):.1%})")
            return False
    
    def get_alert_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return
        
        Returns:
            List of recent alerts
        """
        return self.alert_history[-limit:]
    
    def export_alert_history(self, output_path: str = None) -> pd.DataFrame:
        """
        Export alert history to CSV.
        
        Args:
            output_path: Path to save CSV (optional)
        
        Returns:
            DataFrame with alert history
        """
        if not self.alert_history:
            logger.warning("No alert history to export")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.alert_history)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Exported {len(df)} alerts to {output_path}")
        
        return df


def main():
    """Test alert manager."""
    # Sample prediction
    sample_prediction = {
        'run_id': 12345,
        'run_name': 'CI Pipeline',
        'prediction': 1,
        'prediction_label': 'FAILURE',
        'failure_probability': 0.85,
        'success_probability': 0.15,
        'risk_level': 'CRITICAL',
        'should_alert': True,
        'recommendation': 'Build duration is high. Consider optimizing build steps or caching dependencies.'
    }
    
    # Create alert manager
    manager = AlertManager()
    
    # Process prediction
    alert_sent = manager.process_prediction(sample_prediction)
    
    if alert_sent:
        print("\n✓ Alert sent successfully")
    else:
        print("\n✗ No alert sent")
    
    # Show history
    history = manager.get_alert_history()
    print(f"\nAlert history: {len(history)} alerts")


if __name__ == "__main__":
    main()
