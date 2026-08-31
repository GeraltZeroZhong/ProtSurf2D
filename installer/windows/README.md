# Windows installer

TopoPPI 1.3 provides a bootstrap installer for 64-bit Windows:

```text
TopoPPI-1.3-windows-x86_64-setup.exe
```

The setup program creates a private Python 3.10 environment under
`%LOCALAPPDATA%\TopoPPI`, installs TopoPPI and its scientific dependencies,
copies the matching OptCuts executable, and creates Start Menu shortcuts.

## Install

1. Download the setup executable from the TopoPPI v1.3 GitHub release.
2. Open the downloaded file. The current build is unsigned. If Windows
   SmartScreen blocks it, confirm that it came from the official release page,
   select **More info**, then select **Run anyway**.
3. Keep the setup and PowerShell windows open while the Python environment is
   prepared. A fresh installation commonly takes five to fifteen minutes.
4. Open **TopoPPI GUI** from the Start Menu.

A fresh installation needs access to GitHub, `micro.mamba.pm`, conda-forge, and
PyPI. The bundled OptCuts executable is installed locally. Once setup finishes,
routine analysis of local structures works without a network connection.

The GUI shortcut uses `pythonw.exe`, so regular desktop launches stay free of a
console window. **TopoPPI Command Prompt** opens a persistent terminal with the
private environment and OptCuts ready to use. Run `topoppi --help` there to see
the commands. `TopoPPI CLI.cmd` remains in the installation folder as a direct
wrapper for scripts and automation.

TopoPPI and OptCuts license files are installed under
`%LOCALAPPDATA%\TopoPPI\licenses`.

## Upgrade

Close the GUI and any TopoPPI command prompts, download the installer for the
new release, and run it. Setup updates the existing environment in
`%LOCALAPPDATA%\TopoPPI` and refreshes the launchers. Figures, manifests,
structures, and benchmark results saved elsewhere remain in place.

## Repair an installation

Run the installer for the same release again. It rechecks the Conda environment,
reinstalls the TopoPPI package, and recopies OptCuts.

If the GUI itself cannot start, it shows the location of
`%LOCALAPPDATA%\TopoPPI\gui-startup.log`. That file contains the Python startup
error for the repair report.

If setup stops:

1. Read the final PowerShell message for the failed download or package step.
2. Restore the network connection and run the installer again.
3. If the environment remains incomplete, uninstall TopoPPI, remove
   `%LOCALAPPDATA%\TopoPPI`, and start a fresh installation.

The installer reports a failed bootstrap as a failed setup, including the
PowerShell exit code. A successful completion therefore means the environment
and launchers were created.

## Uninstall

Close the GUI and any TopoPPI command prompts, then use **Settings > Apps >
Installed apps > TopoPPI > Uninstall** or the **Uninstall TopoPPI** Start Menu
shortcut. This removes the private environment, OptCuts, launchers, and
installer files.

TopoPPI leaves analysis output in the folders you selected. The recent-file list
is stored at `%USERPROFILE%\.topoppi\gui_recent.json`; delete the `.topoppi`
folder when you also want to clear that preference data.

## Build locally

Install these tools on a 64-bit Windows development machine:

- Git
- CMake
- Visual Studio 2022 with the **Desktop development with C++** workload
- Inno Setup 6 or later
- network access for the pinned OptCuts source checkout

From the repository root, build OptCuts and the setup executable:

```powershell
.\tools\OptCuts\build_windows_optcuts.ps1 -OutputDir installer\windows
Push-Location installer\windows
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  /DMyAppVersion=1.3 `
  /DMyPackageSpec=topoppi==1.3 `
  TopoPPI.iss
Pop-Location
```

Use the PyPI package spec after TopoPPI 1.3 is available there. A tag archive
also works as `MyPackageSpec` when preparing a release in parallel with PyPI:

```text
https://github.com/GeraltZeroZhong/TopoPPI/archive/refs/tags/v1.3.zip
```

The setup executable is written to `installer/windows/Output/`.

## Release contract

The setup executable embeds:

```text
OptCuts_bin-windows-x86_64.exe
```

When a local builder omits that file, the bootstrap downloads the matching
GitHub release asset and validates it with
`OptCuts_bin-windows-x86_64.exe.sha256`.

The `Windows Installer` GitHub Actions workflow builds and runs the native
OptCuts executable, compiles the setup program with the package version, tests
installation and removal, and uploads the installer, OptCuts executable, and
checksum sidecar to the central `Publish` workflow. Tagged builds install the
source archive from that tag. The release stays in draft form until every
platform artifact and the PyPI upload pass.
