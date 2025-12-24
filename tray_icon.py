"""
タスクトレイアイコンでサーバーを管理
"""
import os
import sys
import subprocess
import threading
import time
import logging
from pathlib import Path
from datetime import datetime

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

# ログ設定
script_dir = Path(__file__).parent
log_file = script_dir / "tray_icon.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # コンソールにも出力（pythonw.exeでは表示されないが、デバッグ用）
    ]
)
logger = logging.getLogger(__name__)

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
        logger.info("サーバーは既に起動しています")
        return
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # venvのPythonを使用
    venv_python = script_dir / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        # venvがない場合は通常のPythonを使用
        python_cmd = sys.executable
        logger.warning("venvが見つかりません。システムのPythonを使用します")
    else:
        python_cmd = str(venv_python)
        logger.info(f"venvのPythonを使用: {python_cmd}")
    
    # ログファイルのパス
    server_log_file = script_dir / "server.log"
    
    try:
        # サーバーを起動（非表示ウィンドウで、ログをファイルに出力）
        with open(server_log_file, 'w', encoding='utf-8') as log_file:
            server_process = subprocess.Popen(
                [python_cmd, "main.py"],
                cwd=script_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # stderrもstdoutに統合
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
        
        # 少し待ってからプロセスの状態を確認
        time.sleep(1)
        
        # プロセスが既に終了している場合（エラーで起動失敗）
        if server_process.poll() is not None:
            is_running = False
            error_code = server_process.returncode
            logger.error(f"サーバーの起動に失敗しました（終了コード: {error_code}）")
            logger.error(f"ログファイルを確認してください: {server_log_file}")
            
            # ログファイルの最後の数行を読み取る
            try:
                with open(server_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_lines = ''.join(lines[-10:])  # 最後の10行
                        logger.error(f"サーバーログの最後の部分:\n{last_lines}")
            except Exception as e:
                logger.error(f"ログファイルの読み取りに失敗: {e}")
            
            server_process = None
            show_error_notification(f"サーバーの起動に失敗しました\nログ: {server_log_file}")
            return
        
        is_running = True
        logger.info(f"サーバーを起動しました（PID: {server_process.pid}）")
        logger.info(f"ログファイル: {server_log_file}")
        
        # サーバーが実際に起動するまで少し待つ（最大10秒）
        logger.info("サーバーの起動を確認中...")
        for i in range(10):
            time.sleep(1)
            try:
                import urllib.request
                response = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
                if response.getcode() == 200:
                    logger.info("サーバーが正常に起動しました")
                    break
            except Exception:
                if i == 9:
                    logger.warning("サーバーの起動確認に失敗しましたが、プロセスは実行中です")
                continue
        
    except Exception as e:
        is_running = False
        logger.error(f"サーバーの起動に失敗しました: {e}", exc_info=True)
        show_error_notification(f"サーバーの起動に失敗しました\nエラー: {str(e)}\nログ: {server_log_file}")

def stop_server():
    """サーバーを停止"""
    global server_process, is_running
    
    if not is_running or server_process is None:
        logger.info("サーバーは既に停止しています")
        return
    
    try:
        logger.info("サーバーを停止中...")
        server_process.terminate()
        # 5秒待ってから強制終了
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("サーバーが正常に終了しませんでした。強制終了します")
            server_process.kill()
        
        server_process = None
        is_running = False
        logger.info("サーバーを停止しました")
    except Exception as e:
        logger.error(f"サーバーの停止に失敗しました: {e}", exc_info=True)

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

def show_error_notification(message: str):
    """エラー通知を表示（Windowsの場合）"""
    if sys.platform == 'win32':
        try:
            # Windowsの標準的な通知方法を使用
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                message,
                "Obsidian MCP Server - エラー",
                0x10 | 0x0  # MB_ICONERROR | MB_OK
            )
        except Exception as e:
            logger.error(f"通知の表示に失敗: {e}")

def check_server_status():
    """サーバーの状態をチェック"""
    global is_running, server_process
    
    if server_process is not None:
        # プロセスが終了しているかチェック
        if server_process.poll() is not None:
            is_running = False
            return_code = server_process.returncode
            logger.warning(f"サーバープロセスが終了しました（終了コード: {return_code}）")
            server_process = None
        else:
            # プロセスは実行中だが、実際にHTTPサーバーが応答するか確認
            try:
                import urllib.request
                response = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
                if response.getcode() == 200:
                    is_running = True
                else:
                    is_running = False
            except Exception:
                # サーバーがまだ起動していない可能性がある
                # プロセスは実行中なので、is_runningはTrueのまま
                pass
    
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
    logger.info("タスクトレイアイコンを表示しました")
    logger.info("右クリックでメニューを表示できます")
    logger.info(f"ログファイル: {log_file}")
    
    # 起動時に自動的にサーバーを起動
    logger.info("サーバーを自動的に起動します...")
    
    icon.run()

if __name__ == "__main__":
    main()

