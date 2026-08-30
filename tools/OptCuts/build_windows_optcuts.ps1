[CmdletBinding()]
param(
    [string]$OutputDir = "release-assets",
    [string]$SourceDir = "",
    [string]$OptCutsRepo = "https://github.com/liminchen/OptCuts.git",
    [string]$OptCutsCommit = "cd2302671af7954f263b0ea93d8419aa943d54be"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"

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
    if (Test-Path $SourceDir) {
        throw "SourceDir exists but is not a Git checkout: $SourceDir"
    }
    Invoke-External "git" @("clone", $OptCutsRepo, $SourceDir)
}

Invoke-External "git" @("-C", $SourceDir, "fetch", "--depth", "1", "origin", $OptCutsCommit)
Invoke-External "git" @("-C", $SourceDir, "checkout", "--detach", $OptCutsCommit)

$CMakeLists = Join-Path $SourceDir "CMakeLists.txt"
$MainCpp = Join-Path $SourceDir "src\main.cpp"
$OptimizerCpp = Join-Path $SourceDir "src\Optimizer.cpp"
$ReproducibilityPatch = Join-Path $PSScriptRoot "reproducible\candidate-validity-cd230267.patch"
$ObjOutputPrecisionPatch = Join-Path $PSScriptRoot "reproducible\obj-output-precision-cd230267.patch"
$StaticStbPatch = Join-Path $PSScriptRoot "reproducible\static-stb-cd230267.patch"
$SparseLocalSolvesPatch = Join-Path $PSScriptRoot "reproducible\sparse-local-solves-cd230267.patch"
$OscillationTolerancePatch = Join-Path $PSScriptRoot "reproducible\oscillation-tolerance-cd230267.patch"
$TopologyCycleAccelerationPatch = Join-Path $PSScriptRoot "reproducible\topology-cycle-acceleration-cd230267.patch"
$Mpl2SparseSolverPatch = Join-Path $PSScriptRoot "reproducible\mpl2-sparse-solver-cd230267.patch"
$ResidueAwarePatch = Join-Path $PSScriptRoot "residue_aware\optcuts-cd230267.patch"
$SourceProvenancePatch = Join-Path $PSScriptRoot "residue_aware\source-vertex-provenance-cd230267.patch"
$FootprintHeader = Join-Path $PSScriptRoot "residue_aware\ResidueFootprintEnergy.hpp"
$FootprintSource = Join-Path $PSScriptRoot "residue_aware\ResidueFootprintEnergy.cpp"
$StbExportHeader = Join-Path $SourceDir "ext\libigl\external\stb_image\igl_stb_image_export.h"
$SortableRowHeader = Join-Path $SourceDir "ext\libigl\include\igl\SortableRow.h"

foreach ($Patch in @($ReproducibilityPatch, $ObjOutputPrecisionPatch, $StaticStbPatch, $SparseLocalSolvesPatch, $OscillationTolerancePatch, $ResidueAwarePatch, $SourceProvenancePatch, $TopologyCycleAccelerationPatch, $Mpl2SparseSolverPatch)) {
    & git -C $SourceDir apply --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        Invoke-External "git" @("-C", $SourceDir, "apply", $Patch)
    }
    else {
        & git -C $SourceDir apply --reverse --check $Patch 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Patch does not match the pinned OptCuts checkout: $Patch"
        }
    }
}
Copy-Item -Force $FootprintHeader (Join-Path $SourceDir "src\ResidueFootprintEnergy.hpp")
Copy-Item -Force $FootprintSource (Join-Path $SourceDir "src\ResidueFootprintEnergy.cpp")

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

// TOPOPPI_WINDOWS_PATCH: stb_image is linked statically into the release
// executable. The vendored source header precedes CMake's generated header,
// so make its visibility macros static-library safe for MSVC.
#define IGL_STB_IMAGE_EXPORT
#define IGL_STB_IMAGE_NO_EXPORT
#define IGL_STB_IMAGE_DEPRECATED __declspec(deprecated)

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

$OptimizerText = Get-Content $OptimizerCpp -Raw
if ($OptimizerText -notmatch "TOPOPPI_WINDOWS_OPTIMIZER_EXTERN_PATCH") {
    $OptimizerText = $OptimizerText -replace "extern const std::string outputFolderPath;", "extern std::string outputFolderPath; // TOPOPPI_WINDOWS_OPTIMIZER_EXTERN_PATCH"
    $OptimizerText = $OptimizerText -replace "extern const bool fractureMode;", "extern bool fractureMode;"
    Set-Content -Path $OptimizerCpp -Value $OptimizerText -Encoding UTF8
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
$DeterministicFlags = "/experimental:deterministic /pathmap:`"$RepoRoot`"=TopoPPI"
Invoke-External "cmake" @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_C_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /MT /Brepro $DeterministicFlags",
    "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /DEIGEN_MPL2_ONLY /MT /Brepro $DeterministicFlags",
    "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /Brepro /DEBUG:NONE"
)
$TbbVersionFile = Join-Path $BuildDir "ext\tbb\version_string.ver"
Set-Content -Path $TbbVersionFile -Encoding ASCII -Value @'
#define __TBB_VERSION_STRINGS(N) \
#N": BUILD_HOST         release-builder" ENDL \
#N": BUILD_OS           Windows" ENDL \
#N": BUILD_KERNEL       generic" ENDL \
#N": BUILD_COMPILER     C++" ENDL \
#N": BUILD_LIBC         system" ENDL \
#N": BUILD_LD           system" ENDL \
#N": BUILD_TARGET       native" ENDL \
#N": BUILD_COMMAND      TopoPPI release build" ENDL

#define __TBB_DATETIME "Unknown"
'@
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
$ArtifactBytes = [System.IO.File]::ReadAllBytes($Artifact)
$ArtifactText = [System.Text.Encoding]::ASCII.GetString($ArtifactBytes) +
    [System.Text.Encoding]::Unicode.GetString($ArtifactBytes)
foreach ($PrivatePath in @($RepoRoot, $SourceDir, $WorkRoot, "D:\a\", "\Users\runner\")) {
    if ($ArtifactText.IndexOf($PrivatePath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        throw "OptCuts artifact contains a private build path: $PrivatePath"
    }
}
foreach ($BlockedMetadata in @("microsoft-standard", "runner\work\")) {
    if ($ArtifactText.IndexOf($BlockedMetadata, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        throw "OptCuts artifact contains runner-specific build metadata: $BlockedMetadata"
    }
}
if ($ArtifactText -match "TBB: BUILD_OS\s+[^\r\n\x00]*\d") {
    throw "OptCuts artifact contains a versioned TBB BUILD_OS value."
}
foreach ($ExpectedMetadata in @(
    "TBB: BUILD_HOST         release-builder",
    "TBB: BUILD_OS           Windows",
    "TBB: BUILD_KERNEL       generic",
    "TBB: BUILD_COMPILER     C++"
)) {
    if ($ArtifactText.IndexOf($ExpectedMetadata, [System.StringComparison]::Ordinal) -lt 0) {
        throw "OptCuts artifact is missing neutral TBB metadata: $ExpectedMetadata"
    }
}
$Hash = (Get-FileHash $Artifact -Algorithm SHA256).Hash.ToLower()
"$Hash  OptCuts_bin-windows-x86_64.exe" | Set-Content "$Artifact.sha256" -Encoding ASCII

$InputMesh = Join-Path $SourceDir "input\bimba_i_f10000.obj"
Push-Location $WorkRoot
try {
    Invoke-External $Artifact @("100", $InputMesh, "0.999", "1", "0", "4.1", "1", "0", "windowsArtifactSmoke")
}
finally {
    Pop-Location
}

Write-Host "Built $Artifact"
Write-Host "SHA256 $Hash"
