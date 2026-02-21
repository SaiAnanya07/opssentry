"""
Log Parser for Different Formats
Parses logs from various sources (HDFS, Spark, GitHub Actions, etc.)
"""
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LogParser:
    """Parses logs from different formats and sources."""
    
    @staticmethod
    def parse_hdfs_log(log_file: Path) -> pd.DataFrame:
        """
        Parse HDFS log file.
        
        HDFS log format example:
        081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906
        
        Args:
            log_file: Path to HDFS log file
        
        Returns:
            DataFrame with parsed log entries
        """
        logger.info(f"Parsing HDFS log: {log_file}")
        
        entries = []
        pattern = r'(\d{6}\s\d{6})\s+(\d+)\s+(\w+)\s+([\w\.\$]+):\s+(.+)'
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(pattern, line)
                if match:
                    timestamp, thread_id, level, component, message = match.groups()
                    entries.append({
                        'timestamp': timestamp,
                        'thread_id': thread_id,
                        'level': level,
                        'component': component,
                        'message': message,
                        'line_number': line_num,
                        'source': 'HDFS'
                    })
                else:  
                    entries.append({
                        'timestamp': None,
                        'thread_id': None,
                        'level': 'UNKNOWN',
                        'component': 'UNKNOWN',
                        'message': line,
                        'line_number': line_num,
                        'source': 'HDFS'
                    })
        
        df = pd.DataFrame(entries)
        logger.info(f"Parsed {len(df)} log entries from HDFS")
        return df
    
    @staticmethod
    def parse_spark_log(log_file: Path) -> pd.DataFrame:
        """
        Parse Spark log file.
        
        Spark log format example:
        15/06/05 18:45:26 INFO SparkContext: Running Spark version 1.3.1
        
        Args:
            log_file: Path to Spark log file
        
        Returns:
            DataFrame with parsed log entries
        """
        logger.info(f"Parsing Spark log: {log_file}")
        
        entries = []
        pattern = r'(\d{2}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2})\s+(\w+)\s+([\w\.\$]+):\s+(.+)'
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(pattern, line)
                if match:
                    timestamp, level, component, message = match.groups()
                    entries.append({
                        'timestamp': timestamp,
                        'level': level,
                        'component': component,
                        'message': message,
                        'line_number': line_num,
                        'source': 'Spark'
                    })
                else:
                    entries.append({
                        'timestamp': None,
                        'level': 'UNKNOWN',
                        'component': 'UNKNOWN',
                        'message': line,
                        'line_number': line_num,
                        'source': 'Spark'
                    })
        
        df = pd.DataFrame(entries)
        logger.info(f"Parsed {len(df)} log entries from Spark")
        return df
    
    @staticmethod
    def parse_github_actions_log(log_file: Path) -> pd.DataFrame:
        """
        Parse GitHub Actions log file.
        
        GitHub Actions log format example:
        2023-01-15T10:30:45.1234567Z ##[group]Run actions/checkout@v3
        
        Args:
            log_file: Path to GitHub Actions log file
        
        Returns:
            DataFrame with parsed log entries
        """
        logger.info(f"Parsing GitHub Actions log: {log_file}")
        
        entries = []
        pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.+)'
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(pattern, line)
                if match:
                    timestamp, message = match.groups()
                    
                    # Extract level from message
                    level = 'INFO'
                    if '##[error]' in message:
                        level = 'ERROR'
                    elif '##[warning]' in message:
                        level = 'WARNING'
                    elif '##[debug]' in message:
                        level = 'DEBUG'
                    
                    entries.append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': message,
                        'line_number': line_num,
                        'source': 'GitHub Actions'
                    })
                else:
                    entries.append({
                        'timestamp': None,
                        'level': 'UNKNOWN',
                        'message': line,
                        'line_number': line_num,
                        'source': 'GitHub Actions'
                    })
        
        df = pd.DataFrame(entries)
        logger.info(f"Parsed {len(df)} log entries from GitHub Actions")
        return df
    
    @staticmethod
    def extract_error_patterns(df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract common error patterns from logs.
        
        Args:
            df: DataFrame with parsed logs
        
        Returns:
            Dict mapping error patterns to their counts
        """
        error_df = df[df['level'].isin(['ERROR', 'FATAL', 'CRITICAL'])]
        
        error_patterns = {}
        for message in error_df['message']:
            # Extract key error indicators
            if 'exception' in message.lower():
                error_patterns['exception'] = error_patterns.get('exception', 0) + 1
            if 'timeout' in message.lower():
                error_patterns['timeout'] = error_patterns.get('timeout', 0) + 1
            if 'failed' in message.lower():
                error_patterns['failed'] = error_patterns.get('failed', 0) + 1
            if 'connection' in message.lower():
                error_patterns['connection'] = error_patterns.get('connection', 0) + 1
            if 'memory' in message.lower():
                error_patterns['memory'] = error_patterns.get('memory', 0) + 1
        
        return error_patterns
    
    @staticmethod
    def parse_log_file(log_file: Path, source_type: str = None) -> pd.DataFrame:
        """
        Auto-detect and parse log file based on source type.
        
        Args:
            log_file: Path to log file
            source_type: Type of log source ('hdfs', 'spark', 'github', etc.)
        
        Returns:
            DataFrame with parsed log entries
        """
        if source_type is None:
            # Auto-detect based on file name or content
            file_name = log_file.name.lower()
            if 'hdfs' in file_name:
                source_type = 'hdfs'
            elif 'spark' in file_name:
                source_type = 'spark'
            elif 'github' in file_name or 'actions' in file_name:
                source_type = 'github'
            else:
                logger.warning(f"Could not auto-detect source type for {log_file}, defaulting to generic parser")
                source_type = 'generic'
        
        parsers = {
            'hdfs': LogParser.parse_hdfs_log,
            'spark': LogParser.parse_spark_log,
            'github': LogParser.parse_github_actions_log,
        }
        
        parser = parsers.get(source_type)
        if parser:
            return parser(log_file)
        else:
            logger.warning(f"No specific parser for {source_type}, using generic parser")
            return LogParser.parse_generic_log(log_file)
    
    @staticmethod
    def parse_generic_log(log_file: Path) -> pd.DataFrame:
        """
        Generic log parser for unstructured logs.
        
        Args:
            log_file: Path to log file
        
        Returns:
            DataFrame with parsed log entries
        """
        logger.info(f"Parsing generic log: {log_file}")
        
        entries = []
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                entries.append({
                    'message': line,
                    'line_number': line_num,
                    'source': 'Generic'
                })
        
        df = pd.DataFrame(entries)
        logger.info(f"Parsed {len(df)} log entries")
        return df
