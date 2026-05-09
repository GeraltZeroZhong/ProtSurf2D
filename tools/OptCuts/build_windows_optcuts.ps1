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
$StbExportHeader = Join-Path $SourceDir "ext\libigl\external\stb_image\igl_stb_image_export.h"
$SortableRowHeader = Join-Path $SourceDir "ext\libigl\include\igl\SortableRow.h"

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

$StbExportText = Get-Content $StbExportHeader -Raw
if ($StbExportText -notmatch "TOPOPPI_WINDOWS_PATCH") {
    Set-Content -Path $StbExportHeader -Encoding UTF8 -Value @'
#ifndef IGL_STB_IMAGE_EXPORT_H
#define IGL_STB_IMAGE_EXPORT_H

// TOPOPPI_WINDOWS_PATCH: the vendored header uses GCC visibility attributes
// even when CMake generates a Windows export header. The source include path
// wins on MSVC, so provide a Windows-safe replacement before configuring.
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef igl_stb_image_EXPORTS
#    define IGL_STB_IMAGE_EXPORT __declspec(dllexport)
#  else
#    define IGL_STB_IMAGE_EXPORT __declspec(dllimport)
#  endif
#  define IGL_STB_IMAGE_NO_EXPORT
#  define IGL_STB_IMAGE_DEPRECATED __declspec(deprecated)
#else
#  define IGL_STB_IMAGE_EXPORT __attribute__((visibility("default")))
#  define IGL_STB_IMAGE_NO_EXPORT __attribute__((visibility("hidden")))
#  define IGL_STB_IMAGE_DEPRECATED __attribute__((__deprecated__))
#endif

#define IGL_STB_IMAGE_DEPRECATED_EXPORT IGL_STB_IMAGE_EXPORT IGL_STB_IMAGE_DEPRECATED
#define IGL_STB_IMAGE_DEPRECATED_NO_EXPORT IGL_STB_IMAGE_NO_EXPORT IGL_STB_IMAGE_DEPRECATED

#endif /* IGL_STB_IMAGE_EXPORT_H */
'@
}

$DoubleMaxLiteral = "1.7976931348623158e+308"
Get-ChildItem -Path (Join-Path $SourceDir "src") -Recurse -Include "*.cpp", "*.hpp", "*.h" | ForEach-Object {
    $SourceText = Get-Content $_.FullName -Raw
    if ($SourceText -match "__DBL_MAX__") {
        $SourceText = $SourceText -replace "__DBL_MAX__", $DoubleMaxLiteral
        Set-Content -Path $_.FullName -Value $SourceText -Encoding UTF8
    }
}

$SortableRowText = Get-Content $SortableRowHeader -Raw
if ($SortableRowText -notmatch "TOPOPPI_WINDOWS_SORTABLE_ROW_PATCH") {
    $SortableRowText = $SortableRowText -replace "const SortableRow<T> & THIS = \*this;", "const SortableRow<T> & self = *this; // TOPOPPI_WINDOWS_SORTABLE_ROW_PATCH"
    $SortableRowText = $SortableRowText -replace "\bTHIS\.", "self."
    Set-Content -Path $SortableRowHeader -Value $SortableRowText -Encoding UTF8
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
