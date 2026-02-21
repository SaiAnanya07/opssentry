# OpsSentry Jenkins CI/CD - Complete Setup Guide

## 🎯 Overview

This guide will walk you through setting up a complete Jenkins CI/CD pipeline for the OpsSentry ML model deployment system.

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Jenkins installed and running (http://localhost:8080)
- ✅ Docker Desktop installed and running
- ✅ Git installed
- ✅ Python 3.12+ installed
- ✅ OpsSentry project cloned/available

---

## 🚀 Step 1: Access Jenkins

1. Open your browser and navigate to: `http://localhost:8080`
2. Login with your Jenkins credentials

---

## 🔌 Step 2: Install Required Plugins

1. Go to: **Manage Jenkins** → **Manage Plugins**
2. Click on **"Available plugins"** tab
3. Search and install the following plugins:

### Required Plugins:
- ✅ **Pipeline** (usually pre-installed)
- ✅ **Git plugin** (usually pre-installed)
- ✅ **Docker Pipeline**
- ✅ **Docker plugin**
- ✅ **Python Plugin**
- ✅ **Credentials Binding Plugin**
- ✅ **Email Extension Plugin**
- ✅ **Workspace Cleanup Plugin**

4. Click **"Install without restart"**
5. Check **"Restart Jenkins when installation is complete"**
6. Wait for Jenkins to restart

---

## 🔑 Step 3: Configure Credentials

### GitHub Credentials (if using private repository)

1. Go to: **Manage Jenkins** → **Manage Credentials**
2. Click on **(global)** domain
3. Click **"Add Credentials"**
4. Configure:
   - **Kind**: Username with password
   - **Username**: Your GitHub username
   - **Password**: Your GitHub Personal Access Token
   - **ID**: `github-credentials`
   - **Description**: GitHub Access Token
5. Click **"Create"**

### Docker Hub Credentials (optional, for pushing images)

1. Click **"Add Credentials"** again
2. Configure:
   - **Kind**: Username with password
   - **Username**: Your Docker Hub username
   - **Password**: Your Docker Hub password
   - **ID**: `dockerhub-credentials`
   - **Description**: Docker Hub Credentials
3. Click **"Create"**

---

## 📝 Step 4: Create Jenkins Pipeline Job

### Create New Pipeline

1. From Jenkins dashboard, click **"New Item"**
2. Enter item name: `OpsSentry-CI-CD`
3. Select **"Pipeline"**
4. Click **"OK"**

### Configure Pipeline

1. **General Section:**
   - ✅ Check **"Discard old builds"**
   - Strategy: Log Rotation
   - Days to keep builds: `30`
   - Max # of builds to keep: `10`

2. **Build Triggers:**
   - ✅ Check **"Poll SCM"** (optional, for automatic builds)
   - Schedule: `H/15 * * * *` (checks every 15 minutes)
   - OR configure GitHub webhook for instant triggers

3. **Pipeline Section:**
   - **Definition**: Pipeline script from SCM
   - **SCM**: Git
   - **Repository URL**: `file:///c:/Users/yedit/OneDrive/Desktop/final_yr_project/opssentry`
     - Or your Git repository URL if using remote
   - **Credentials**: Select `github-credentials` (if needed)
   - **Branch Specifier**: `*/main` or `*/master`
   - **Script Path**: `Jenkinsfile`

4. Click **"Save"**

---

## ⚙️ Step 5: Configure Pipeline Parameters

The Jenkinsfile includes these parameters (automatically configured):

- **TUNE_HYPERPARAMETERS**: Enable hyperparameter tuning (slower but better models)
- **SKIP_SMOTE**: Skip SMOTE for class imbalance handling
- **DEPLOY_TO_PRODUCTION**: Deploy to production after successful build
- **LOG_LEVEL**: Logging level (INFO/DEBUG/WARNING)

These will appear when you click **"Build with Parameters"**

---

## 🏃 Step 6: Run Your First Build

1. Go to your pipeline: **OpsSentry-CI-CD**
2. Click **"Build with Parameters"**
3. Leave default parameters or customize:
   - TUNE_HYPERPARAMETERS: `false` (for faster first build)
   - SKIP_SMOTE: `false`
   - DEPLOY_TO_PRODUCTION: `false`
   - LOG_LEVEL: `INFO`
4. Click **"Build"**

### Monitor Build Progress

1. Click on the build number (e.g., `#1`)
2. Click **"Console Output"** to see real-time logs
3. Or use **"Blue Ocean"** view for better visualization (if installed)

---

## 📊 Step 7: Understand Pipeline Stages

Your pipeline will execute these stages:

1. **Checkout** - Get source code from repository
2. **Setup Environment** - Create Python virtual environment and install dependencies
3. **Validate Data** - Check datasets are available and valid
4. **Train Models** - Train ML models (Random Forest, XGBoost, Gradient Boosting)
5. **Evaluate Models** - Validate model performance against thresholds
6. **Run Tests** - Execute integration tests
7. **Build Docker Image** - Create Docker container image
8. **Test Docker Image** - Verify Docker image works correctly
9. **Deploy to Staging** - Deploy using docker-compose
10. **Health Check** - Verify application is running and healthy
11. **Deploy to Production** - (Optional) Deploy to production if parameter is set

---

## 🎨 Step 8: View Build Artifacts

After a successful build:

1. Go to your build (e.g., `#1`)
2. Click **"Build Artifacts"**
3. You'll find:
   - Trained models (`*.pkl`)
   - Performance plots (`*.png`)
   - Log files (`*.log`)

---

## 📧 Step 9: Configure Email Notifications (Optional)

### Configure Email Extension

1. Go to: **Manage Jenkins** → **Configure System**
2. Scroll to **"Extended E-mail Notification"**
3. Configure:
   - **SMTP server**: `smtp.gmail.com` (for Gmail)
   - **SMTP Port**: `587`
   - **Use SMTP Authentication**: ✅ Check
   - **User Name**: Your email
   - **Password**: Your app password
   - **Use TLS**: ✅ Check
4. Set **"Default Recipients"**: Your email address
5. Click **"Save"**

---

## 🔄 Step 10: Configure GitHub Webhook (Optional)

For automatic builds on code push:

1. Go to your GitHub repository
2. Click **"Settings"** → **"Webhooks"** → **"Add webhook"**
3. Configure:
   - **Payload URL**: `http://your-jenkins-url:8080/github-webhook/`
   - **Content type**: `application/json`
   - **Events**: Just the push event
4. Click **"Add webhook"**

---

## 🧪 Testing Your Pipeline

### Test 1: Verify Build Success

```powershell
# Check if Jenkins can access your project
cd c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry
git status
```

### Test 2: Verify Docker

```powershell
# Check Docker is running
docker ps

# Check if OpsSentry image was built
docker images | findstr opssentry
```

### Test 3: Verify Deployment

```powershell
# Check if application is running
curl http://localhost:5000/api/health
```

---

## 📁 Pipeline File Structure

```
opssentry/
├── Jenkinsfile                 # Main pipeline configuration
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Local deployment configuration
├── .dockerignore              # Docker build exclusions
├── scripts/
│   ├── validate_model.py      # Model validation script
│   ├── health_check.py        # Health check script
│   └── train_model.py         # Model training script
├── tests/
│   └── test_integration.py    # Integration tests
└── requirements.txt           # Python dependencies
```

---

## 🔧 Troubleshooting

### Build Fails at "Setup Environment"

**Problem**: Python or pip not found

**Solution**:
```powershell
# Verify Python is in PATH
python --version

# Add Python to Jenkins PATH
# Manage Jenkins → Configure System → Global properties
# Add Environment Variable: PATH = C:\Python312;C:\Python312\Scripts;$PATH
```

### Build Fails at "Build Docker Image"

**Problem**: Docker not running or not accessible

**Solution**:
```powershell
# Start Docker Desktop
# Verify Docker is running
docker ps

# Restart Docker service if needed
```

### Build Fails at "Train Models"

**Problem**: No training data available

**Solution**:
```powershell
# Generate sample data first
cd c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry
python create_sample_data.py

# Or collect real data
python scripts\data_collector.py
python scripts\preprocess.py
```

### Health Check Fails

**Problem**: Application not responding

**Solution**:
```powershell
# Check if port 5000 is available
netstat -ano | findstr :5000

# Check Docker container logs
docker logs opssentry-app

# Restart the application
docker-compose restart
```

---

## 🎯 Best Practices

1. **Start Simple**: First build without hyperparameter tuning
2. **Monitor Resources**: Watch CPU/memory usage during builds
3. **Archive Artifacts**: Keep important models and metrics
4. **Use Parameters**: Customize builds without editing Jenkinsfile
5. **Test Locally**: Test Docker builds locally before Jenkins
6. **Version Control**: Keep Jenkinsfile in version control
7. **Secure Credentials**: Never hardcode passwords in Jenkinsfile

---

## 📊 Monitoring Your Pipeline

### View Build History

1. Go to **OpsSentry-CI-CD** dashboard
2. See build history with success/failure indicators
3. Click on any build to see details

### View Trends

1. Install **"Build Monitor Plugin"** for dashboard view
2. Use **"Blue Ocean"** for modern pipeline visualization
3. Check **"Test Results Trend"** for test history

---

## 🚀 Next Steps

After successful pipeline setup:

1. ✅ Run a test build
2. ✅ Verify all stages pass
3. ✅ Check artifacts are archived
4. ✅ Test the deployed application
5. ✅ Configure notifications
6. ✅ Set up automatic builds (webhook)
7. ✅ Document any custom configurations

---

## 📞 Need Help?

- **Jenkins Documentation**: https://www.jenkins.io/doc/
- **Docker Documentation**: https://docs.docker.com/
- **OpsSentry README**: See project README.md

---

**Congratulations! Your Jenkins CI/CD pipeline for OpsSentry is now set up!** 🎉
