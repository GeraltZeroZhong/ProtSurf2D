[CmdletBinding()]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\TopoPPI",
    [string]$Version = "1.2",
    [string]$PackageSpec = "",
    [string]$MicromambaUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    $InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
    $BinDir = Join-Path $InstallDir "bin"
    $EnvDir = Join-Path $InstallDir "env"
    $RootPrefix = Join-Path $InstallDir "mamba-root"
    $TempDir = Join-Path $InstallDir "tmp"
    $OptCutsDir = Join-Path $InstallDir "bin"
    $Micromamba = Join-Path $BinDir "micromamba.exe"
    $BundledOptCuts = Join-Path $InstallDir "installer\assets\OptCuts_bin-windows-x86_64.exe"

    New-Item -ItemType Directory -Force -Path $InstallDir, $BinDir, $TempDir | Out-Null

    if (!(Test-Path $Micromamba)) {
        Write-Step "Downloading micromamba"
        $Archive = Join-Path $TempDir "micromamba.tar.bz2"
        $ExtractDir = Join-Path $TempDir "micromamba"
        Remove-Item -Recurse -Force $ExtractDir -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null

        Invoke-WebRequest -Uri $MicromambaUrl -OutFile $Archive
        Invoke-External "tar.exe" @("-xjf", $Archive, "-C", $ExtractDir)

        $ExtractedMicromamba = Get-ChildItem -Path $ExtractDir -Recurse -Filter "micromamba.exe" |
            Select-Object -First 1
        if ($null -eq $ExtractedMicromamba) {
            throw "micromamba.exe was not found in the downloaded archive."
        }
        Copy-Item -Force $ExtractedMicromamba.FullName $Micromamba
    }

    $env:MAMBA_ROOT_PREFIX = $RootPrefix
    $Packages = @(
        "python=3.10",
        "tk",
        "igl=2.6.*",
        "numpy>=1.21",
        "scipy",
        "biopython",
        "scikit-image",
        "matplotlib",
        "trimesh",
        "networkx",
        "pillow",
        "rtree",
        "shapely",
        "openbabel",
        "mdanalysis",
        "rdkit",
        "psutil",
        "tqdm",
        "meshio",
        "pip"
    )

    if (Test-Path $EnvDir) {
        Write-Step "Updating TopoPPI Conda environment"
        Invoke-External $Micromamba (@("install", "-y", "-p", $EnvDir, "-c", "conda-forge") + $Packages)
    }
    else {
        Write-Step "Creating TopoPPI Conda environment"
        Invoke-External $Micromamba (@("create", "-y", "-p", $EnvDir, "-c", "conda-forge") + $Packages)
    }

    $Python = Join-Path $EnvDir "python.exe"
    $OptCutsInstaller = Join-Path $EnvDir "Scripts\topoppi-install-optcuts.exe"
    if (!(Test-Path $Python)) {
        throw "Python was not installed at $Python."
    }

    $ResolvedPackageSpec = $PackageSpec
    if ([string]::IsNullOrWhiteSpace($ResolvedPackageSpec)) {
        $ResolvedPackageSpec = "topoppi==$Version"
    }

    Write-Step "Installing TopoPPI from $ResolvedPackageSpec"
    Invoke-External $Python @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-External $Python @("-m", "pip", "install", "prolif>=2.0")
    Invoke-External $Python @("-m", "pip", "install", "--no-deps", $ResolvedPackageSpec)

    if (!(Test-Path $OptCutsInstaller)) {
        throw "topoppi-install-optcuts was not installed at $OptCutsInstaller."
    }

    Write-Step "Installing Windows OptCuts artifact"
    if (Test-Path $BundledOptCuts) {
        $BundledOptCutsUri = ([System.Uri]$BundledOptCuts).AbsoluteUri
        $BundledOptCutsSha = (Get-FileHash $BundledOptCuts -Algorithm SHA256).Hash.ToLower()
        Invoke-External $OptCutsInstaller @(
            "--platform", "windows-x86_64",
            "--url", $BundledOptCutsUri,
            "--checksum", $BundledOptCutsSha,
            "--install-dir", $OptCutsDir,
            "--force"
        )
    }
    else {
        Invoke-External $OptCutsInstaller @("--platform", "windows-x86_64", "--install-dir", $OptCutsDir, "--force")
    }

    $OptCutsExe = Join-Path $OptCutsDir "OptCuts_bin.exe"
    if (!(Test-Path $OptCutsExe)) {
        throw "OptCuts was not installed at $OptCutsExe."
    }

    Write-Step "Writing launchers"
    $GuiLauncher = @"
@echo off
set "TOPOPPI_HOME=$InstallDir"
set "TOPOPPI_OPTCUTS_BIN=$OptCutsExe"
"$EnvDir\Scripts\topoppi-gui.exe" %*
"@
    Set-Content -Path (Join-Path $InstallDir "TopoPPI GUI.cmd") -Value $GuiLauncher -Encoding ASCII

    $CliLauncher = @"
@echo off
set "TOPOPPI_HOME=$InstallDir"
set "TOPOPPI_OPTCUTS_BIN=$OptCutsExe"
"$EnvDir\Scripts\topoppi.exe" %*
"@
    Set-Content -Path (Join-Path $InstallDir "TopoPPI CLI.cmd") -Value $CliLauncher -Encoding ASCII

    Write-Step "TopoPPI installation finished"
}
catch {
    Write-Error "TopoPPI installation failed: $($_.Exception.Message)"
    Write-Error "If this failed while installing OptCuts, confirm the GitHub release includes OptCuts_bin-windows-x86_64.exe and its .sha256 sidecar."
    exit 1
}
