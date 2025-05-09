FROM python:3.9-slim-bullseye

ENV XDG_RUNTIME_DIR=/tmp

# 日本語フォント（japanize_matplotlib 用など）やビルドに必要なツールを追加
RUN apt-get update && apt-get install -y \
        build-essential \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack-dev \
        fonts-ipafont-gothic \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python パッケージを先に requirements.txt でまとめてインストール
WORKDIR /workspaces
COPY requirements.txt /workspaces/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー（必要に応じて調整）
#COPY . /app
