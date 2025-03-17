import os
import sys
import webbrowser
import threading
import time
from main import app


def open_browser(port):
    """지정된 시간 후에 브라우저를 엽니다"""
    time.sleep(1.5)  # 서버가 시작될 때까지 잠시 대기
    url = f"http://localhost:{port}"
    webbrowser.open(url)


if __name__ == '__main__':
    # 실행 파일 경로를 기준으로 상대 경로 설정
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        application_path = os.path.dirname(sys.executable)
    else:
        # 일반 Python 스크립트로 실행된 경우
        application_path = os.path.dirname(os.path.abspath(__file__))

    os.chdir(application_path)

    # 포트 설정 (기본값 8050)
    port = 8050

    # 별도 스레드에서 브라우저 열기
    threading.Thread(target=open_browser, args=(port,)).start()

    # 서버 실행 (open_browser 매개변수 제거)
    app.run_server(debug=False, port=port, host='localhost')
