"""Serve the v7 viewer on its own port.

Same samples and API as the other viewers. "/" is this page only:

    python3 viewer/serve_v7.py   -> http://127.0.0.1:8777/

Port comes from V7_PORT (default 8777). 8765 is left alone.
"""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

import server

V7_INDEX = "/static/v7/index.html"
HOST = "0.0.0.0"
PORT = int(os.environ.get("V7_PORT", "8777"))


class V7Handler(server.Handler):
    def do_GET(self) -> None:
        if urlparse(self.path).path in ("/", "/index.html"):
            self.path = V7_INDEX
        super().do_GET()


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), V7Handler)
    print(f"Ego viewer http://127.0.0.1:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
