# OpsSentry Jenkins CI/CD - Installation Guide

This guide will walk you through installing all the required software for running OpsSentry with Jenkins CI/CD automation.

## 📋 Prerequisites Checklist

Before you begin, you'll need to install the following:

- [ ] Java Development Kit (JDK) 11 or 17
- [ ] Jenkins
- [ ] Docker Desktop
- [ ] Git (already installed)
- [ ] Python 3.8+ (already installed)

---

## 1️⃣ Install Java Development Kit (JDK)

Jenkins requires Java to run. We'll install OpenJDK 17 (LTS version).

### Windows Installation

**Option A: Using Chocolatey (Recommended)**

```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install OpenJDK 17
choco install openjdk17 -y
```

**Option B: Manual Installation**

1. Download OpenJDK 17 from: https://adoptium.net/temurin/releases/
2. Choose: **Windows x64**, **JDK 17 (LTS)**, **.msi installer**
3. Run the installer and follow the wizard
4. Check "Set JAVA_HOME variable" and "Add to PATH" during installation

**Verify Installation:**

```powershell
java -version
# Should show: openjdk version "17.x.x"
```

---

## 2️⃣ Install Jenkins

### Windows Installation

**Option A: Using Chocolatey (Recommended)**

```powershell
# Install Jenkins
choco install jenkins -y
```

**Option B: Manual Installation**

1. Download Jenkins for Windows: https://www.jenkins.io/download/
2. Choose **Windows** → Download the `.msi` installer
3. Run the installer:
   - Accept default installation directory: `C:\Program Files\Jenkins`
   - Jenkins will run as a Windows service
   - Default port: **8080**
4. The installer will automatically start Jenkins

**Access Jenkins:**

1. Open browser and navigate to: `http://localhost:8080`
2. Get the initial admin password:

```powershell
# Read the initial admin password
Get-Content "C:\Program Files\Jenkins\secrets\initialAdminPassword"
```

3. Copy the password and paste it in the browser
4. Click **"Install suggested plugins"**
5. Create your first admin user:
   - Username: `admin` (or your choice)
   - Password: (choose a strong password)
   - Full name: Your name
   - Email: Your email
6. Keep the default Jenkins URL: `http://localhost:8080/`
7. Click **"Start using Jenkins"**

---

## 3️⃣ Install Required Jenkins Plugins

After Jenkins is set up, install these essential plugins:

1. Go to **Dashboard** → **Manage Jenkins** → **Manage Plugins**
2. Click on **"Available plugins"** tab
3. Search and select the following plugins:

### Required Plugins:

- [ ] **Pipeline** (should already be installed)
- [ ] **Git plugin** (should already be installed)
- [ ] **Docker Pipeline**
- [ ] **Docker plugin**
- [ ] **Python Plugin**
- [ ] **Credentials Binding Plugin**
- [ ] **Email Extension Plugin** (for notifications)
- [ ] **Slack Notification Plugin** (optional, for Slack alerts)
- [ ] **Blue Ocean** (optional, for better UI)
- [ ] **Workspace Cleanup Plugin**

4. Click **"Install without restart"**
5. Check **"Restart Jenkins when installation is complete and no jobs are running"**

---

## 4️⃣ Install Docker Desktop

Docker is required for containerizing your OpsSentry application.

### Windows Installation

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click **"Download for Windows"**

2. **System Requirements:**
   - Windows 10/11 64-bit: Pro, Enterprise, or Education
   - WSL 2 backend (Windows Subsystem for Linux)
   - Virtualization must be enabled in BIOS

3. **Install Docker Desktop:**
   - Run the installer: `Docker Desktop Installer.exe`
   - Follow the installation wizard
   - **Important:** Check **"Use WSL 2 instead of Hyper-V"** (recommended)
   - Restart your computer when prompted

4. **Start Docker Desktop:**
   - Launch Docker Desktop from Start Menu
   - Accept the Docker Subscription Service Agreement
   - Wait for Docker to start (whale icon in system tray should be steady)

**Verify Installation:**

```powershell
# Check Docker version
docker --version
# Should show: Docker version 24.x.x or higher

# Check Docker is running
docker run hello-world
# Should download and run a test container
```

---

## 5️⃣ Configure Jenkins for Docker

### Enable Docker in Jenkins

1. Go to **Dashboard** → **Manage Jenkins** → **Global Tool Configuration**

2. **Add Docker Installation:**
   - Scroll to **Docker** section
   - Click **"Add Docker"**
   - Name: `docker`
   - Check **"Install automatically"**
   - Choose **"Download from docker.com"**
   - Docker version: **latest**
   - Click **"Save"**

3. **Configure Docker Cloud (Optional):**
   - Go to **Dashboard** → **Manage Jenkins** → **Manage Nodes and Clouds**
   - Click **"Configure Clouds"** → **"Add a new cloud"** → **"Docker"**
   - Docker Host URI: `tcp://localhost:2375` (Windows) or `unix:///var/run/docker.sock` (Linux)

---

## 6️⃣ Configure Jenkins Credentials

You'll need to set up credentials for GitHub and Docker Hub.

### Add GitHub Credentials

1. Go to **Dashboard** → **Manage Jenkins** → **Manage Credentials**
2. Click **(global)** domain
3. Click **"Add Credentials"**
4. Configure:
   - Kind: **Username with password**
   - Username: Your GitHub username
   - Password: Your GitHub Personal Access Token (create one at https://github.com/settings/tokens)
   - ID: `github-credentials`
   - Description: `GitHub Access Token`
5. Click **"Create"**

### Add Docker Hub Credentials (Optional)

1. Click **"Add Credentials"** again
2. Configure:
   - Kind: **Username with password**
   - Username: Your Docker Hub username
   - Password: Your Docker Hub password
   - ID: `dockerhub-credentials`
   - Description: `Docker Hub Credentials`
3. Click **"Create"**

---

## 7️⃣ Verify All Installations

Run this verification script to check all installations:

```powershell
# Create verification script
Write-Host "=== OpsSentry Installation Verification ===" -ForegroundColor Cyan

# Check Java
Write-Host "`n1. Checking Java..." -ForegroundColor Yellow
java -version
if ($LASTEXITCODE -eq 0) { Write-Host "✓ Java installed" -ForegroundColor Green } else { Write-Host "✗ Java not found" -ForegroundColor Red }

# Check Jenkins
Write-Host "`n2. Checking Jenkins..." -ForegroundColor Yellow
$jenkinsService = Get-Service -Name "Jenkins" -ErrorAction SilentlyContinue
if ($jenkinsService -and $jenkinsService.Status -eq "Running") {
    Write-Host "✓ Jenkins service is running" -ForegroundColor Green
} else {
    Write-Host "✗ Jenkins service not running" -ForegroundColor Red
}

# Check Docker
Write-Host "`n3. Checking Docker..." -ForegroundColor Yellow
docker --version
if ($LASTEXITCODE -eq 0) { Write-Host "✓ Docker installed" -ForegroundColor Green } else { Write-Host "✗ Docker not found" -ForegroundColor Red }

# Check Docker is running
Write-Host "`n4. Checking Docker daemon..." -ForegroundColor Yellow
docker ps > $null 2>&1
if ($LASTEXITCODE -eq 0) { Write-Host "✓ Docker daemon is running" -ForegroundColor Green } else { Write-Host "✗ Docker daemon not running" -ForegroundColor Red }

# Check Python
Write-Host "`n5. Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -eq 0) { Write-Host "✓ Python installed" -ForegroundColor Green } else { Write-Host "✗ Python not found" -ForegroundColor Red }

# Check Git
Write-Host "`n6. Checking Git..." -ForegroundColor Yellow
git --version
if ($LASTEXITCODE -eq 0) { Write-Host "✓ Git installed" -ForegroundColor Green } else { Write-Host "✗ Git not found" -ForegroundColor Red }

Write-Host "`n=== Verification Complete ===" -ForegroundColor Cyan
```

---

## 8️⃣ Quick Installation Commands (Summary)

If you want to install everything quickly using Chocolatey:

```powershell
# Run PowerShell as Administrator

# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install all required software
choco install openjdk17 jenkins docker-desktop -y

# Restart computer
Restart-Computer
```

---

## 🎯 Next Steps

After completing all installations:

1. ✅ Verify all installations using the verification script above
2. ✅ Access Jenkins at `http://localhost:8080`
3. ✅ Install required Jenkins plugins
4. ✅ Configure Jenkins credentials
5. ✅ Proceed to create your Jenkins pipeline

**Ready to continue?** Once all installations are complete, we'll create the Jenkinsfile and Docker configuration for your OpsSentry project!

---

## 🔧 Troubleshooting

### Jenkins won't start
- Check if port 8080 is already in use
- Restart Jenkins service: `Restart-Service Jenkins`
- Check logs: `C:\Program Files\Jenkins\jenkins.log`

### Docker won't start
- Enable virtualization in BIOS
- Enable WSL 2: `wsl --install`
- Restart Docker Desktop

### Java not found
- Make sure JAVA_HOME is set: `$env:JAVA_HOME`
- Add Java to PATH manually if needed

### Port conflicts
- Change Jenkins port: Edit `C:\Program Files\Jenkins\jenkins.xml`
- Change Docker port in Docker Desktop settings

---

## 📞 Need Help?

If you encounter any issues during installation, check:
- Jenkins documentation: https://www.jenkins.io/doc/
- Docker documentation: https://docs.docker.com/
- OpsSentry project README
