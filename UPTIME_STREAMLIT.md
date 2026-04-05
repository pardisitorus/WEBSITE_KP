# Uptime Streamlit

File ini menjelaskan setup agar app Streamlit Anda dijaga oleh "pengunjung bayangan" dari GitHub Actions.

## 1. Secret yang wajib diisi

Di GitHub repo `pardisitorus/WEBSITE_KP`:

1. Buka `Settings` -> `Secrets and variables` -> `Actions`.
2. Tambahkan salah satu ini:
   - Name: `STREAMLIT_APP_URL`
   - Value: URL app Streamlit Anda (contoh: `https://nama-app.streamlit.app`)

`Secret` atau `Variable` sama-sama bisa dipakai. Karena ini hanya URL publik, `Variable` biasanya sudah cukup.

## 2. Workflow yang sudah tersedia

File workflow:
- `.github/workflows/streamlit-keepalive.yml`
- `.github/scripts/streamlit-keepalive.mjs`

Perilaku:
- Jalan otomatis setiap 30 menit.
- Membuka app dengan browser headless.
- Jika halaman sleep muncul, workflow mencoba klik tombol wake.
- Gagal jika app tetap sleep/error, agar terlihat di tab `Actions`.

## 3. Catatan penting

- Ini mendekati konsep "1 pengunjung bayangan", tetapi tetap bergantung pada limit platform Streamlit dan GitHub Actions.
- Scheduled workflow GitHub bisa nonaktif bila repo lama tidak ada aktivitas. Jika itu terjadi, aktifkan lagi dari tab `Actions`.
- Streamlit Community Cloud tetap memiliki limit platform (sleep/resource restart), jadi tidak ada jaminan 100% "never down" selamanya.
- Jika butuh SLA always-on yang ketat, pindah ke hosting non-sleep (server berbayar) adalah opsi yang tepat.
