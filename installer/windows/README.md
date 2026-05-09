# Windows Installer

This directory contains the Inno Setup bootstrap installer for Windows x86-64.

The installer is intentionally a bootstrapper. It downloads `micromamba`, creates
an isolated Python 3.10 environment under `%LOCALAPPDATA%\TopoPPI`, installs the
requested `topoppi` version from PyPI, installs the matching Windows OptCuts
release artifact, and creates Start Menu launchers.

## Build Locally

Install Inno Setup 6 on Windows, then run:

```powershell
.\tools\OptCuts\build_windows_optcuts.ps1 -OutputDir installer\windows
Push-Location installer\windows
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  /DMyAppVersion=1.2 `
  /DMyPackageSpec=topoppi==1.2 `
  TopoPPI.iss
Pop-Location
```

The output is written to `installer/windows/Output/`.

## Required Release Artifacts

For a fully working installer, the setup executable should include:

```text
OptCuts_bin-windows-x86_64.exe
OptCuts_bin-windows-x86_64.exe.sha256
```

The GitHub Actions workflow builds and embeds this OptCuts executable before
compiling the setup executable. The bootstrap installer falls back to the
matching GitHub release artifact only when the bundled executable is absent.

Manual workflow builds can compile a setup executable that installs a GitHub
source archive instead of a PyPI version by passing `MyPackageSpec` at compile
time. Tagged release builds should use a PyPI package spec for the same version
as the tag.
