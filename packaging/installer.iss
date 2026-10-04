; Inno Setup script — AI Live Translator (per-user install, no admin rights needed).
; Build: "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /DAppVersion=1.0.0 packaging\installer.iss
; Input: build\dist\AI Live Translator\  (PyInstaller output). Output: build\installer\

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "AI Live Translator"
#define AppNameAr "المترجم الفوري"
#define AppExe "AI Live Translator.exe"

[Setup]
AppId={{6F2B9C1E-7D4A-4C1B-9E3F-2A8D5B7C9E41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppNameAr} {#AppVersion}
AppPublisher=AI Live Translator
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
OutputDir=..\build\installer
OutputBaseFilename=AI-Live-Translator-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
arabic.DesktopIcon=إنشاء اختصار على سطح المكتب
english.DesktopIcon=Create a desktop shortcut
arabic.OpenVBCable=افتح صفحة تحميل الميكروفون الوهمي VB-CABLE (مطلوب للاجتماعات — مجاني)
english.OpenVBCable=Open the VB-CABLE virtual microphone download page (needed for meetings — free)
arabic.OpenWebView2=افتح صفحة تحميل Microsoft Edge WebView2 (مطلوب لعرض واجهة البرنامج)
english.OpenWebView2=Open the Microsoft Edge WebView2 download page (needed to show the app window)
arabic.LaunchApp=تشغيل المترجم الفوري الآن
english.LaunchApp=Launch AI Live Translator now
arabic.DeleteModels=هل تريد أيضًا حذف نماذج الذكاء الاصطناعي التي تم تنزيلها (عدة جيجابايت) وإعداداتك؟%n%nاختر «لا» إذا كنت ستعيد تثبيت البرنامج لاحقًا.
english.DeleteModels=Also delete the downloaded AI models (several GB) and your settings?%n%nChoose "No" if you plan to reinstall later.

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"

[Files]
Source: "..\build\dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppNameAr}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppNameAr}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "https://developer.microsoft.com/microsoft-edge/webview2/"; Description: "{cm:OpenWebView2}"; Flags: postinstall shellexec skipifsilent; Check: not WebView2Installed
Filename: "https://vb-audio.com/Cable/"; Description: "{cm:OpenVBCable}"; Flags: postinstall shellexec skipifsilent; Check: not VBCableInstalled
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchApp}"; Flags: postinstall nowait skipifsilent

[Code]
function WebView2Installed: Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version)
         or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version);
  Result := Result and (Version <> '') and (Version <> '0.0.0.0');
end;

function VBCableInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\drivers\vbaudio_cable64_win10.sys'))
         or FileExists(ExpandConstant('{sys}\drivers\vbaudio_cable64_win7.sys'))
         or RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\VB-Audio\Cable')
         or RegKeyExists(HKLM, 'SOFTWARE\VB-Audio\Cable');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if SuppressibleMsgBox(CustomMessage('DeleteModels'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
    begin
      DelTree(ExpandConstant('{localappdata}\{#AppName}'), True, True, True);
      DelTree(ExpandConstant('{userappdata}\{#AppName}'), True, True, True);
    end;
end;
