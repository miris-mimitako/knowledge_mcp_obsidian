"""
タスクトレイアイコンでサーバーを管理
"""
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("pystrayとPillowがインストールされていません。")
    print("以下のコマンドでインストールしてください:")
    print("pip install pystray pillow")
    sys.exit(1)

# グローバル変数
server_process = None
server_thread = None
is_running = False
icon = None

def create_icon_image():
    """タスクトレイアイコン用の画像を作成"""
    # シンプルなアイコン画像を作成（16x16）
    image = Image.new('RGB', (64, 64), color='white')
    draw = ImageDraw.Draw(image)
    
    # 円を描画（サーバーアイコンの代わり）
    draw.ellipse([16, 16, 48, 48], fill='#667eea', outline='#764ba2', width=2)
    # 中央に「O」を描画
    draw.text((24, 20), 'O', fill='white')
    
    return image

def start_server():
    """サーバーを起動"""
    global server_process, is_running
    
    if is_running:
        return
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # venvのPythonを使用
    venv_python = script_dir / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        # venvがない場合は通常のPythonを使用
        python_cmd = sys.executable
    else:
        python_cmd = str(venv_python)
    
    try:
        # サーバーを起動（非表示ウィンドウで）
        server_process = subprocess.Popen(
            [python_cmd, "main.py"],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        is_running = True
        print("サーバーを起動しました")
    except Exception as e:
        print(f"サーバーの起動に失敗しました: {e}")

def stop_server():
    """サーバーを停止"""
    global server_process, is_running
    
    if not is_running or server_process is None:
        return
    
    try:
        server_process.terminate()
        # 5秒待ってから強制終了
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        
        server_process = None
        is_running = False
        print("サーバーを停止しました")
    except Exception as e:
        print(f"サーバーの停止に失敗しました: {e}")

def open_browser():
    """ブラウザでサーバーを開く"""
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")

def open_docs():
    """APIドキュメントを開く"""
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000/docs")

def on_start_clicked(icon, item):
    """起動メニューがクリックされたとき"""
    start_server()
    update_menu()

def on_stop_clicked(icon, item):
    """停止メニューがクリックされたとき"""
    stop_server()
    update_menu()

def on_open_clicked(icon, item):
    """ブラウザで開くメニューがクリックされたとき"""
    open_browser()

def on_docs_clicked(icon, item):
    """APIドキュメントメニューがクリックされたとき"""
    open_docs()

def on_quit_clicked(icon, item):
    """終了メニューがクリックされたとき"""
    stop_server()
    icon.stop()

def check_server_status():
    """サーバーの状態をチェック"""
    global is_running, server_process
    
    if server_process is not None:
        # プロセスが終了しているかチェック
        if server_process.poll() is not None:
            is_running = False
            server_process = None
    
    return is_running

def create_menu():
    """タスクトレイメニューを作成"""
    status = check_server_status()
    status_text = "実行中" if status else "停止中"
    
    return pystray.Menu(
        pystray.MenuItem(f"状態: {status_text}", lambda: None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("サーバー起動", on_start_clicked, enabled=lambda item: not status),
        pystray.MenuItem("サーバー停止", on_stop_clicked, enabled=lambda item: status),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("ブラウザで開く", on_open_clicked, enabled=lambda item: status),
        pystray.MenuItem("APIドキュメント", on_docs_clicked, enabled=lambda item: status),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("終了", on_quit_clicked)
    )

def update_menu():
    """メニューを更新"""
    global icon
    if icon:
        icon.menu = create_menu()

def menu_update_thread():
    """メニューを定期的に更新するスレッド"""
    while True:
        time.sleep(2)  # 2秒ごとに更新
        if icon:
            update_menu()

def main():
    """メイン処理"""
    global icon
    
    # アイコン画像を作成
    image = create_icon_image()
    
    # タスクトレイアイコンを作成
    icon = pystray.Icon(
        "Obsidian MCP Server",
        image,
        "Obsidian MCP Server",
        create_menu()
    )
    
    # メニュー更新スレッドを開始
    update_thread = threading.Thread(target=menu_update_thread, daemon=True)
    update_thread.start()
    
    # 起動時に自動的にサーバーを起動
    start_server()
    
    # アイコンを表示
    print("タスクトレイアイコンを表示しました")
    print("右クリックでメニューを表示できます")
    print("サーバーは自動的に起動しました")
    icon.run()

if __name__ == "__main__":
    main()

