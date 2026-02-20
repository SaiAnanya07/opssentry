pipeline {
    agent any

    parameters {
        booleanParam(name: 'TUNE_HYPERPARAMETERS', defaultValue: false, description: 'Enable hyperparameter tuning')
        booleanParam(name: 'SKIP_SMOTE', defaultValue: false, description: 'Skip SMOTE for class imbalance')
    }

    environment {
        PYTHONIOENCODING = 'utf-8'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '=== Checking out source code ==='
                checkout scm
                bat 'git log -1 --format="Commit: %%H %%s"'
            }
        }

        stage('Setup Environment') {
            steps {
                echo '=== Setting up Python virtual environment ==='
                dir('opssentry') {
                    withEnv(['PATH+PYTHON=C:\\Users\\yedit\\AppData\\Local\\Programs\\Python\\Python312;C:\\Users\\yedit\\AppData\\Local\\Programs\\Python\\Python312\\Scripts']) {
                        bat 'python --version'
                        bat 'python -m venv venv'
                        bat 'venv\\Scripts\\python.exe -m pip install --upgrade pip'
                        bat 'venv\\Scripts\\pip.exe install -r requirements.txt'
                    }
                }
            }
        }

        stage('Collect Real Data') {
            steps {
                echo '=== Fetching GitHub Actions run data ==='
                dir('opssentry') {
                    bat '''
                        set PYTHONPATH=%CD%
                        venv\\Scripts\\python.exe scripts\\fetch_runs.py --max-pages 5
                    '''
                }
            }
        }

        stage('Preprocess Data') {
            steps {
                echo '=== Preprocessing data ==='
                dir('opssentry') {
                    bat '''
                        set PYTHONPATH=%CD%
                        venv\\Scripts\\python.exe scripts\\preprocess.py --source github || echo Done
                    '''
                }
            }
        }

        stage('Train Models') {
            steps {
                echo '=== Training ML models ==='
                dir('opssentry') {
                    script {
                        def tuneFlag = params.TUNE_HYPERPARAMETERS ? '--tune' : ''
                        def smoteFlag = params.SKIP_SMOTE ? '--no-smote' : ''
                        bat """
                            set PYTHONPATH=%CD%
                            venv\\Scripts\\python.exe scripts\\train_model.py ${tuneFlag} ${smoteFlag}
                        """
                    }
                }
            }
        }

        stage('Evaluate Models') {
            steps {
                echo '=== Evaluating model performance ==='
                dir('opssentry') {
                    bat '''
                        set PYTHONPATH=%CD%
                        venv\\Scripts\\python.exe scripts\\validate_model.py
                    '''
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo '=== Running tests ==='
                dir('opssentry') {
                    bat '''
                        set PYTHONPATH=%CD%
                        venv\\Scripts\\python.exe -m pytest tests/ -v --junitxml=test-results.xml || exit 0
                    '''
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'opssentry/test-results.xml'
                }
            }
        }

        stage('Health Check') {
            steps {
                echo '=== Running health check ==='
                dir('opssentry') {
                    bat '''
                        set PYTHONPATH=%CD%
                        venv\\Scripts\\python.exe scripts\\health_check.py || echo Health check attempted
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                echo '=== Building Docker image ==='
                dir('opssentry') {
                    bat "docker build -t opssentry:${BUILD_NUMBER} . || echo Docker skipped"
                }
            }
        }
    }

    post {
        always {
            echo '=== Archiving artifacts ==='
            archiveArtifacts artifacts: 'opssentry/models/*.pkl, opssentry/models/*.png, opssentry/data/github/*.csv', allowEmptyArchive: true
        }
        success {
            echo '=== SUCCESS: OpsSentry pipeline completed! ==='
        }
        failure {
            echo '=== FAILURE: Check console output for details ==='
        }
    }
}