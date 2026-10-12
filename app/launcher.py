"""Khởi động web tool và tự mở trình duyệt (dùng cho start.bat / shortcut trên Desktop).

    python -m app.launcher            # cổng mặc định 8000 (tự đổi nếu đang bận)
    python -m app.launcher --no-browser
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import urllib.request
import webbrowser


def free_port(start: int = 8000, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"Không tìm được cổng trống từ {start} đến {start + tries - 1}")


def _open_when_ready(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "api/engines", timeout=2):
                webbrowser.open(url)
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Khởi động web tool dịch PDF khoa học")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    import uvicorn

    port = free_port(args.port)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}/"
    print("=" * 60)
    print(f"  Dịch PDF khoa học Anh → Việt đang chạy tại: {url}")
    print("  Giữ cửa sổ này mở khi dùng. Đóng cửa sổ (hoặc Ctrl+C) để tắt.")
    print("=" * 60)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app.main:app", host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
