import os

target = "test_dir"
try:
    if not os.path.exists(target):
        os.makedirs(target)
        print(f"Created {target}")
    else:
        files = os.listdir(target)
        print(f"Files in {target}: {files}")
except OSError as e:
    print(f"OS error: {e}")
