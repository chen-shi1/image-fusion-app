FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装 Octave 和编译依赖
RUN apt-get update && apt-get install -y \
    octave \
    liboctave-dev \
    liboctave9 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码
COPY . .

# 确保 Octave 可用
ENV OCTAVE_EXECUTABLE=/usr/bin/octave

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
