import os

result = os.popen("fuser -v /dev/nvidia*").read()
results = result.split()
for pid in results:
    if pid.isdigit():  # 检查字符串是否为数字
        os.system(f"kill -9 {int(pid)}")