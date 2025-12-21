@echo off
chcp 65001 >nul
echo ====================================
echo Obsidian MCP Server 起動スクリプト
echo ====================================
echo.

REM スクリプトのディレクトリに移動
cd /d %~dp0

REM venvが存在しない場合は作成
if not exist "venv" (
    echo venv環境を作成中...
    python -m venv venv
    if errorlevel 1 (
        echo エラー: venvの作成に失敗しました
        pause
        exit /b 1
    )
    echo venv環境を作成しました
    echo.
)

REM venvを有効化
echo venv環境を有効化中...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo エラー: venvの有効化に失敗しました
    pause
    exit /b 1
)

REM 依存関係のインストール
echo 依存関係をインストール中...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo エラー: 依存関係のインストールに失敗しました
    pause
    exit /b 1
)
echo.

REM サーバー起動
echo ====================================
echo サーバーを起動します...
echo ====================================
echo アクセス: http://127.0.0.1:8000
echo ドキュメント: http://127.0.0.1:8000/docs
echo.
echo 停止する場合は Ctrl+C を押してください
echo.

python main.py

pause

