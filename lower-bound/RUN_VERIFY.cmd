@echo off
setlocal
cd /d "%~dp0"
python verify_hashes.py
if errorlevel 1 exit /b 1
python src\verify_clean.py --root . --out results\verification.json
if errorlevel 1 exit /b 1
python src\verify_physical.py --certificate certificates\FPLUS\certificate.json --oracle chronological --output results\physical_chronological.json
if errorlevel 1 exit /b 1
python src\verify_physical.py --certificate certificates\FPLUS\certificate.json --oracle displacement --output results\physical_displacement.json
if errorlevel 1 exit /b 1
echo ALL_EXACT_VERIFICATIONS_PASS
