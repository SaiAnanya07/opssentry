"""
Recommendation Engine
Generates actionable recommendations based on failure patterns and feature analysis.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RecommendationEngine:
    """Generates recommendations based on prediction and feature analysis."""
    
    @staticmethod
    def analyze_build_duration(build_duration: float) -> List[str]:
        """
        Analyze build duration and provide recommendations.
        
        Args:
            build_duration: Build duration in seconds
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if build_duration > 900:  # > 15 minutes
            recommendations.append(
                "⏱️ Build duration is very high (>15 min). "
                "Consider: 1) Parallelizing jobs, 2) Caching dependencies, "
                "3) Optimizing test execution, 4) Using incremental builds."
            )
        elif build_duration > 600:  # > 10 minutes
            recommendations.append(
                "⏱️ Build duration is high (>10 min). "
                "Review build steps for optimization opportunities."
            )
        
        return recommendations
    
    @staticmethod
    def analyze_test_failures(num_failed_tests: int, num_total_tests: int) -> List[str]:
        """
        Analyze test failures and provide recommendations.
        
        Args:
            num_failed_tests: Number of failed tests
            num_total_tests: Total number of tests
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if num_failed_tests > 0:
            failure_rate = num_failed_tests / max(num_total_tests, 1)
            
            if failure_rate > 0.5:
                recommendations.append(
                    f"❌ High test failure rate ({failure_rate:.1%}). "
                    "This indicates significant issues. "
                    "Review recent code changes and fix failing tests before proceeding."
                )
            elif failure_rate > 0.2:
                recommendations.append(
                    f"⚠️ Moderate test failure rate ({failure_rate:.1%}). "
                    "Investigate failing tests and ensure adequate test coverage."
                )
            else:
                recommendations.append(
                    f"⚠️ {num_failed_tests} test(s) failing. "
                    "Review test logs and fix issues."
                )
        
        return recommendations
    
    @staticmethod
    def analyze_historical_failures(failure_rate: float, total_failures: int) -> List[str]:
        """
        Analyze historical failure patterns.
        
        Args:
            failure_rate: Recent failure rate (0-1)
            total_failures: Total number of historical failures
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if failure_rate > 0.7:
            recommendations.append(
                f"📊 Very high historical failure rate ({failure_rate:.1%}). "
                "This pipeline is unstable. Consider: "
                "1) Reviewing pipeline configuration, "
                "2) Improving test reliability, "
                "3) Addressing flaky tests."
            )
        elif failure_rate > 0.4:
            recommendations.append(
                f"📊 High historical failure rate ({failure_rate:.1%}). "
                "Investigate common failure patterns and address root causes."
            )
        
        if total_failures > 50:
            recommendations.append(
                f"📈 Pipeline has failed {total_failures} times historically. "
                "Consider implementing preventive measures and better error handling."
            )
        
        return recommendations
    
    @staticmethod
    def analyze_timing(hour_of_day: int, is_weekend: bool) -> List[str]:
        """
        Analyze deployment timing.
        
        Args:
            hour_of_day: Hour of day (0-23)
            is_weekend: Whether it's a weekend
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if is_weekend:
            recommendations.append(
                "📅 Weekend deployment detected. "
                "Ensure: 1) Adequate monitoring is in place, "
                "2) On-call support is available, "
                "3) Rollback plan is ready."
            )
        
        if hour_of_day < 6 or hour_of_day > 22:
            recommendations.append(
                f"🌙 Off-hours deployment ({hour_of_day}:00). "
                "Ensure proper monitoring and support coverage."
            )
        
        return recommendations
    
    @staticmethod
    def analyze_errors(num_errors: int, error_types: List[str]) -> List[str]:
        """
        Analyze error patterns.
        
        Args:
            num_errors: Number of errors
            error_types: List of error types
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if num_errors > 10:
            recommendations.append(
                f"🔴 High number of errors ({num_errors}). "
                "Review logs immediately and address critical errors."
            )
        elif num_errors > 0:
            recommendations.append(
                f"⚠️ {num_errors} error(s) detected. "
                "Check logs for error details."
            )
        
        # Analyze specific error types
        if error_types:
            if 'timeout' in error_types:
                recommendations.append(
                    "⏰ Timeout errors detected. "
                    "Consider: 1) Increasing timeout limits, "
                    "2) Optimizing slow operations, "
                    "3) Checking network connectivity."
                )
            
            if 'memory' in error_types:
                recommendations.append(
                    "💾 Memory errors detected. "
                    "Consider: 1) Increasing memory limits, "
                    "2) Optimizing memory usage, "
                    "3) Checking for memory leaks."
                )
            
            if 'connection' in error_types:
                recommendations.append(
                    "🔌 Connection errors detected. "
                    "Check: 1) Network connectivity, "
                    "2) Service availability, "
                    "3) Authentication credentials."
                )
        
        return recommendations
    
    @staticmethod
    def generate_comprehensive_recommendations(run_data: Dict[str, Any],
                                              prediction: Dict[str, Any]) -> str:
        """
        Generate comprehensive recommendations based on all available data.
        
        Args:
            run_data: Pipeline run data
            prediction: Prediction result
        
        Returns:
            Formatted recommendation string
        """
        all_recommendations = []
        
        # Priority message based on risk level
        risk_level = prediction.get('risk_level', 'UNKNOWN')
        if risk_level == 'CRITICAL':
            all_recommendations.append(
                "🚨 CRITICAL: This pipeline has a very high probability of failure. "
                "Immediate action required!"
            )
        elif risk_level == 'WARNING':
            all_recommendations.append(
                "⚠️ WARNING: This pipeline has elevated failure risk. "
                "Review the following recommendations carefully."
            )
        
        # Analyze different aspects
        if 'build_duration' in run_data:
            all_recommendations.extend(
                RecommendationEngine.analyze_build_duration(run_data['build_duration'])
            )
        
        if 'num_tests_failed' in run_data and 'num_tests_run' in run_data:
            all_recommendations.extend(
                RecommendationEngine.analyze_test_failures(
                    run_data['num_tests_failed'],
                    run_data['num_tests_run']
                )
            )
        
        if 'failure_rate' in run_data:
            total_failures = run_data.get('total_failures', 0)
            all_recommendations.extend(
                RecommendationEngine.analyze_historical_failures(
                    run_data['failure_rate'],
                    total_failures
                )
            )
        
        if 'hour_of_day' in run_data:
            all_recommendations.extend(
                RecommendationEngine.analyze_timing(
                    run_data['hour_of_day'],
                    run_data.get('is_weekend', False)
                )
            )
        
        if 'num_errors' in run_data:
            all_recommendations.extend(
                RecommendationEngine.analyze_errors(
                    run_data['num_errors'],
                    run_data.get('error_types', [])
                )
            )
        
        # General recommendations
        if not all_recommendations:
            all_recommendations.append(
                "✅ Monitor pipeline execution closely. "
                "Check logs for any warnings or unexpected behavior."
            )
        
        # Format recommendations
        if len(all_recommendations) == 1:
            return all_recommendations[0]
        else:
            return "\n\n".join(f"{i+1}. {rec}" for i, rec in enumerate(all_recommendations))
    
    @staticmethod
    def get_similar_failure_patterns(run_data: Dict[str, Any],
                                     historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find similar historical failures for pattern analysis.
        
        Args:
            run_data: Current pipeline run data
            historical_data: DataFrame with historical runs
        
        Returns:
            List of similar failure cases
        """
        if historical_data.empty:
            return []
        
        # Filter to failed runs only
        failed_runs = historical_data[historical_data['failed'] == 1]
        
        if failed_runs.empty:
            return []
        
        # Find similar runs based on key features
        similar_runs = []
        
        # Simple similarity: same workflow name and similar build duration
        if 'name' in run_data and 'name' in failed_runs.columns:
            same_workflow = failed_runs[failed_runs['name'] == run_data.get('name')]
            
            if not same_workflow.empty:
                # Get most recent failures
                if 'created_at' in same_workflow.columns:
                    same_workflow = same_workflow.sort_values('created_at', ascending=False)
                
                similar_runs = same_workflow.head(5).to_dict('records')
        
        return similar_runs


def main():
    """Test recommendation engine."""
    # Sample run data
    run_data = {
        'name': 'CI Pipeline',
        'build_duration': 750.0,
        'num_tests_failed': 3,
        'num_tests_run': 50,
        'failure_rate': 0.6,
        'total_failures': 45,
        'hour_of_day': 23,
        'is_weekend': True,
        'num_errors': 5,
        'error_types': ['timeout', 'connection']
    }
    
    prediction = {
        'risk_level': 'CRITICAL',
        'failure_probability': 0.85
    }
    
    engine = RecommendationEngine()
    recommendations = engine.generate_comprehensive_recommendations(run_data, prediction)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(recommendations)
    print("=" * 60)


if __name__ == "__main__":
    main()
