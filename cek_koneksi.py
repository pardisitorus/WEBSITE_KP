import ee

print("1. Sedang mencoba menghubungi Google Earth Engine...")
try:
    ee.Initialize()
    print("✅ SUKSES! Laptop sudah terhubung ke Satelit.")
    print("   Silakan jalankan app.py sekarang.")
except Exception as e:
    print("❌ GAGAL KONEKSI.")
    print("   Penyebab:", e)
    print("\nSOLUSI: Buka CMD, ketik: earthengine authenticate")