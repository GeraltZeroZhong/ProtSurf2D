[CmdletBinding()]
param(
    [string]$OutputDir = "release-assets",
    [string]$SourceDir = "",
    [string]$OptCutsRepo = "https://github.com/liminchen/OptCuts.git",
    [string]$OptCutsCommit = "cd2302671af7954f263b0ea93d8419aa943d54be",
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

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

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputDir = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
$WorkRoot = Join-Path $RepoRoot ".optcuts-windows-build"
$BuildDir = Join-Path $WorkRoot "build"

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $WorkRoot "OptCuts"
}
$SourceDir = [System.IO.Path]::GetFullPath($SourceDir)

New-Item -ItemType Directory -Force -Path $OutputDir, $WorkRoot | Out-Null

if (!(Test-Path (Join-Path $SourceDir ".git"))) {
    Remove-Item -Recurse -Force $SourceDir -ErrorAction SilentlyContinue
    Invoke-External "git" @("clone", $OptCutsRepo, $SourceDir)
}

Invoke-External "git" @("-C", $SourceDir, "fetch", "--depth", "1", "origin", $OptCutsCommit)
Invoke-External "git" @("-C", $SourceDir, "checkout", "--detach", $OptCutsCommit)

$CMakeLists = Join-Path $SourceDir "CMakeLists.txt"
$MainCpp = Join-Path $SourceDir "src\main.cpp"

$CMakeText = Get-Content $CMakeLists -Raw
if ($CMakeText -notmatch "TOPOPPI_WINDOWS_PATCH") {
    $CMakeText += @'

# TOPOPPI_WINDOWS_PATCH: build the release artifact on GitHub Actions Windows runners.
if(MSVC)
  target_compile_definitions(${PROJECT_NAME}_bin PRIVATE _USE_MATH_DEFINES NOMINMAX)
  target_compile_options(${PROJECT_NAME}_bin PRIVATE /bigobj /permissive-)
endif()
'@
    Set-Content -Path $CMakeLists -Value $CMakeText -Encoding UTF8
}

$MainText = Get-Content $MainCpp -Raw
if ($MainText -notmatch "TOPOPPI_WINDOWS_PATCH") {
    $MainText = $MainText -replace "#include <sys/stat.h> // for mkdir", @"
#include <sys/stat.h> // for mkdir

// TOPOPPI_WINDOWS_PATCH: MSVC exposes mkdir as _mkdir with one argument.
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif
"@
    Set-Content -Path $MainCpp -Value $MainText -Encoding UTF8
}

Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
Invoke-External "cmake" @("-S", $SourceDir, "-B", $BuildDir, "-G", "Visual Studio 17 2022", "-A", "x64")
Invoke-External "cmake" @("--build", $BuildDir, "--config", "Release", "--parallel")

$BuiltExeCandidates = @(
    (Join-Path $BuildDir "Release\OptCuts_bin.exe"),
    (Join-Path $BuildDir "OptCuts_bin.exe")
)
$BuiltExe = $BuiltExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($null -eq $BuiltExe) {
    throw "OptCuts_bin.exe was not produced by the Windows build."
}

$Artifact = Join-Path $OutputDir "OptCuts_bin-windows-x86_64.exe"
Copy-Item -Force $BuiltExe $Artifact
$Hash = (Get-FileHash $Artifact -Algorithm SHA256).Hash.ToLower()
"$Hash  OptCuts_bin-windows-x86_64.exe" | Set-Content "$Artifact.sha256" -Encoding ASCII

if (!$SkipSmoke) {
    $InputMesh = Join-Path $SourceDir "input\bimba_i_f10000.obj"
    if (Test-Path $InputMesh) {
        Push-Location $WorkRoot
        try {
            Invoke-External $Artifact @("100", $InputMesh, "0.999", "1", "0", "4.1", "1", "0", "windowsArtifactSmoke")
        }
        finally {
            Pop-Location
        }
    }
}

Write-Host "Built $Artifact"
Write-Host "SHA256 $Hash"
