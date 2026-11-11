; Inno Setup script. Build after PyInstaller:  iscc installer\noize.iss
#define AppName "NOIZE"
#define AppVersion "0.1.0"

[Setup]
AppId={{6F0B5C34-9E2A-4B70-8C1D-4E5A1B7D0A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NOIZE
DefaultDirName={autopf}\NOIZE
DefaultGroupName=NOIZE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\noize\assets\NOIZE_app_icon.ico
UninstallDisplayIcon={app}\NOIZE.exe
OutputDir=..\dist
OutputBaseFilename=NOIZE-Setup-{#AppVersion}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "..\dist\NOIZE\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\NOIZE"; Filename: "{app}\NOIZE.exe"; IconFilename: "{app}\NOIZE.exe"
Name: "{autodesktop}\NOIZE"; Filename: "{app}\NOIZE.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\NOIZE.exe"; Description: "Launch NOIZE"; Flags: nowait postinstall skipifsilent
