[CmdletBinding()]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\TopoPPI"
)

$ErrorActionPreference = "Stop"

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$GeneratedPaths = @(
    (Join-Path $InstallDir "env"),
    (Join-Path $InstallDir "mamba-root"),
    (Join-Path $InstallDir "tmp"),
    (Join-Path $InstallDir "bin"),
    (Join-Path $InstallDir "TopoPPI GUI.cmd"),
    (Join-Path $InstallDir "TopoPPI CLI.cmd"),
    (Join-Path $InstallDir "TopoPPI Command Prompt.cmd"),
    (Join-Path $InstallDir "gui-startup.log")
)

foreach ($Path in $GeneratedPaths) {
    Remove-Item -Recurse -Force $Path -ErrorAction SilentlyContinue
}
