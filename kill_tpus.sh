lsof -w /lib/libtpu.so | grep "python" |  awk '{print $2}' | xargs -r kill -9
