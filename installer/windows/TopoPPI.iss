#define MyAppName "TopoPPI"
#ifndef MyAppVersion
#define MyAppVersion "1.2"
#endif
#ifndef MyPackageSpec
#define MyPackageSpec ""
#endif

[Setup]
AppId={{4F4E6672-FF5E-43AF-9DC5-5E3E64CE4FE3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=TopoPPI contributors
AppPublisherURL=https://github.com/GeraltZeroZhong/TopoPPI
AppSupportURL=https://github.com/GeraltZeroZhong/TopoPPI/issues
AppUpdatesURL=https://github.com/GeraltZeroZhong/TopoPPI/releases
DefaultDirName={localappdata}\TopoPPI
DefaultGroupName=TopoPPI
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=TopoPPI-{#MyAppVersion}-windows-x86_64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
SetupIconFile=..\..\src\topoppi\assets\topoppi.ico
UninstallDisplayIcon={app}\installer\assets\topoppi.ico

[Files]
Source: "install_topoppi.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "uninstall_topoppi.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "..\..\src\topoppi\assets\topoppi.ico"; DestDir: "{app}\installer\assets"; Flags: ignoreversion
#ifexist "OptCuts_bin-windows-x86_64.exe"
Source: "OptCuts_bin-windows-x86_64.exe"; DestDir: "{app}\installer\assets"; Flags: ignoreversion
#endif
#ifexist "OptCuts_bin-windows-x86_64.exe.sha256"
Source: "OptCuts_bin-windows-x86_64.exe.sha256"; DestDir: "{app}\installer\assets"; Flags: ignoreversion
#endif

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\install_topoppi.ps1"" -InstallDir ""{app}"" -Version ""{#MyAppVersion}"" -PackageSpec ""{#MyPackageSpec}"""; Description: "Install TopoPPI environment"; Flags: waituntilterminated

[Icons]
Name: "{group}\TopoPPI GUI"; Filename: "{app}\TopoPPI GUI.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\installer\assets\topoppi.ico"
Name: "{group}\TopoPPI CLI"; Filename: "{app}\TopoPPI CLI.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\installer\assets\topoppi.ico"
Name: "{group}\Uninstall TopoPPI"; Filename: "{uninstallexe}"; IconFilename: "{app}\installer\assets\topoppi.ico"

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\uninstall_topoppi.ps1"" -InstallDir ""{app}"""; Flags: waituntilterminated runhidden
