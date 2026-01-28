; ADX FPM - Inno Setup Installer Script
; Requires: Inno Setup 6.x (https://jrsoftware.org/isinfo.php)
; Compile with: ISCC.exe installer.iss

#define MyAppName "ADX FPM"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "ADX Studio"
#define MyAppURL "https://github.com/adhadimohd/ADX-Studio"
#define MyAppExeName "launcher.bat"

[Setup]
AppId={{A3D2F1E0-8B4C-4D5A-9E6F-7A8B9C0D1E2F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=ADX_FPM_Setup_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Application source code
Source: "run_web.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "install.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "launcher.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "runv.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "runv.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: ".env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "CLAUDE.md"; DestDir: "{app}"; Flags: ignoreversion

; Python package
Source: "adxfpm\*.py"; DestDir: "{app}\adxfpm"; Flags: ignoreversion recursesubdirs
Source: "adxfpm\templates\*"; DestDir: "{app}\adxfpm\templates"; Flags: ignoreversion recursesubdirs

; ML models
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion

[Dirs]
Name: "{app}\input"
Name: "{app}\outputs"
Name: "{app}\uploads"
Name: "{app}\logs"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Post-install: create venv and install dependencies
Filename: "python"; Parameters: "install.py"; WorkingDir: "{app}"; StatusMsg: "Installing Python dependencies (this may take several minutes)..."; Flags: runhidden waituntilterminated
; Optionally launch the app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\outputs"
Type: filesandordirs; Name: "{app}\uploads"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\input"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\adxfpm\__pycache__"
Type: files; Name: "{app}\.env"
