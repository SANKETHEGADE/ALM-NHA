import os
import sys
import time

def test_speed():
    test_dir = os.path.join(os.path.dirname(__file__), "test_out")
    os.makedirs(test_dir, exist_ok=True)
    file_path = os.path.join(test_dir, "test.bin")
    
    t0 = time.time()
    chunk = b"\x00\x01\x02\x03" * (256 * 1024) # 1 MB chunk
    with open(file_path, "wb") as f:
        for _ in range(100): # 100 MB
            f.write(chunk)
    t1 = time.time()
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Wrote {size_mb:.2f} MB in {t1 - t0:.2f} seconds ({size_mb / (t1 - t0):.2f} MB/s)")
    
    # Cleanup
    os.remove(file_path)
    os.rmdir(test_dir)

if __name__ == "__main__":
    test_speed()
