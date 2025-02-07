@echo off
call venv\Scripts\activate.bat
:: Tampilkan teks loading awal (tanpa newline)
<nul set /p=Memuat aplikasi, harap tunggu

:: Animasi loading horizontal: menampilkan titik-titik di samping
for /l %%x in (1,1,10) do (
    <nul set /p=.
    ping -n 2 127.0.0.1 >nul
)
echo.
python app.py
pause
