# Cài đặt "Dịch PDF khoa học Anh -> Việt" trên Windows (chạy qua install.bat).
# Việc làm: tìm/cài Python 3.12 -> tạo môi trường .venv -> cài thư viện -> chuyển mô hình
# VinAI sang CTranslate2 int8 -> tạo shortcut trên Desktop -> mở web tool.
param(
    [switch]$NoShortcut,   # không tạo shortcut Desktop
    [switch]$NoStart,      # không mở tool sau khi cài
    [switch]$SkipModels,   # không tải/chuyển mô hình (dùng cho kiểm thử CI)
    [switch]$KeepTorch,    # giữ torch sau khi chuyển mô hình (mặc định gỡ để tiết kiệm ~1 GB)
    [string]$Models = "vinai"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Log = Join-Path $Root "install.log"
Start-Transcript -Path $Log -Append | Out-Null

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host ""; Write-Host "LỖI: $msg" -ForegroundColor Red
    Write-Host "Chi tiết trong file: $Log"
    Stop-Transcript | Out-Null
    exit 1
}

function Find-Python {
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }
    try {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -match "^3\.(1[0-3])$") {
            return (& python -c "import sys; print(sys.executable)").Trim()
        }
    } catch {}
    return $null
}

Step "Kiểm tra Python"
$Py = Find-Python
if (-not $Py) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Step "Cài Python 3.12 bằng winget (cần Internet)"
        winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        $Py = Find-Python
    }
    if (-not $Py) {
        Start-Process "https://www.python.org/downloads/windows/"
        Fail "Chưa có Python 3.10–3.12. Hãy cài Python (tích 'Add python.exe to PATH') rồi chạy lại install.bat."
    }
}
Write-Host "Dùng Python: $Py"

Step "Tạo môi trường riêng (.venv)"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    & $Py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Không tạo được .venv" }
}

Step "Cài thư viện (lần đầu mất vài phút)"
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r requirements.txt -r requirements-ml.txt
if ($LASTEXITCODE -ne 0) { Fail "Cài thư viện thất bại (kiểm tra kết nối Internet)" }

# Mô hình đã chuyển đổi sẵn (chạy lại bộ cài) → không cần tải torch / chuyển lại
$Pending = @()
foreach ($m in $Models.Split(",")) {
    $m = $m.Trim()
    $ready = $false
    try {
        & $VenvPy -c "import sys; from app.translators import get_engine; sys.exit(0 if get_engine('$m')._ct2_ready() else 1)"
        $ready = ($LASTEXITCODE -eq 0)
    } catch {}
    if ($ready) { Write-Host "Mô hình '$m' đã sẵn sàng, bỏ qua." } else { $Pending += $m }
}
if ($Pending.Count -eq 0) { $SkipModels = $true }
$Models = $Pending -join ","

if (-not $SkipModels) {
    Step "Cài torch bản CPU (chỉ dùng để chuyển đổi mô hình một lần)"
    & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Fail "Cài torch thất bại" }

    foreach ($m in $Models.Split(",")) {
        Step "Tải và chuyển mô hình '$m' sang CTranslate2 int8 (tải ~1,5 GB lần đầu)"
        & $VenvPy scripts\convert_ct2.py $m.Trim()
        if ($LASTEXITCODE -ne 0) { Fail "Chuyển mô hình '$m' thất bại" }
    }

    if (-not $KeepTorch) {
        Step "Gỡ torch (không cần khi dịch nữa)"
        & $VenvPy -m pip uninstall -y torch | Out-Null
    }
}

Step "Kiểm tra cài đặt"
& $VenvPy -m app.doctor
if ($LASTEXITCODE -ne 0 -and -not $SkipModels) { Write-Host "Có cảnh báo ở trên — tool vẫn có thể chạy." -ForegroundColor Yellow }

if (-not $NoShortcut) {
    Step "Tạo shortcut trên Desktop"
    try {
        $Desktop = [Environment]::GetFolderPath("Desktop")
        # WScript.Shell không ghi được tên file Unicode ("Dịch" -> "D?ch") nên tạo với tên
        # không dấu rồi đổi tên bằng PowerShell (hỗ trợ Unicode).
        $Tmp = Join-Path $Desktop "PDF-Translate-tmp.lnk"
        $Shell = New-Object -ComObject WScript.Shell
        $Lnk = $Shell.CreateShortcut($Tmp)
        $Lnk.TargetPath = Join-Path $Root "windows\start.bat"
        $Lnk.WorkingDirectory = $Root
        $Lnk.IconLocation = "$env:SystemRoot\System32\shell32.dll,70"
        $Lnk.Description = "Dich PDF khoa hoc Anh - Viet (chay tren may)"
        $Lnk.Save()
        $Final = Join-Path $Desktop "Dịch PDF khoa học.lnk"
        try {
            Move-Item -LiteralPath $Tmp -Destination $Final -Force
        } catch {
            $Final = Join-Path $Desktop "Dich PDF khoa hoc.lnk"
            Move-Item -LiteralPath $Tmp -Destination $Final -Force
        }
        Write-Host "Đã tạo shortcut: $Final"
    } catch {
        Write-Host "Không tạo được shortcut ($($_.Exception.Message)). Mở tool bằng windows\start.bat" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "XONG! Mở tool bằng shortcut 'Dịch PDF khoa học' trên Desktop hoặc windows\start.bat" -ForegroundColor Green
Stop-Transcript | Out-Null
if (-not $NoStart) { Start-Process (Join-Path $Root "windows\start.bat") }
