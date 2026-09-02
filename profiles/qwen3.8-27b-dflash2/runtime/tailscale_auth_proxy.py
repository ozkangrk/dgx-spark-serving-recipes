#!/usr/bin/env python3
"""Bearer-auth reverse proxy bound only to the host's Tailscale IPv4 address."""
from __future__ import annotations

import argparse
import hmac
import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
MAX_BODY = 32 * 1024 * 1024


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "QwenTailscaleProxy/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        # Never log headers or request bodies; standard status/path only.
        super().log_message(fmt, *args)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.api_key}"  # type: ignore[attr-defined]
        return hmac.compare_digest(supplied, expected)

    def _deny(self) -> None:
        body = b'{"error":"unauthorized"}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        if not self._authorized():
            self._deny()
            return
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > MAX_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in HOP_HEADERS and k.lower() not in {"host", "authorization", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection("127.0.0.1", 8888, timeout=900)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS and key.lower() != "content-length":
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not self.wfile.closed:
                self.close_connection = True
        finally:
            conn.close()

    do_GET = _proxy
    do_POST = _proxy
    do_DELETE = _proxy
    do_PUT = _proxy
    do_PATCH = _proxy
    do_OPTIONS = _proxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--key-file", required=True)
    args = parser.parse_args()
    key_path = Path(args.key_file)
    api_key = key_path.read_text(encoding="utf-8").strip()
    if len(api_key) < 32:
        raise SystemExit("API key file is missing or too short")
    server = ThreadingHTTPServer((args.bind, args.port), ProxyHandler)
    server.api_key = api_key  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
