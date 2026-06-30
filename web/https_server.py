"""
HTTPS server for the PBC Verifier web app.

Run from the web/ directory:
    python https_server.py

Before first use, create a local self-signed certificate and keep it untracked:

    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout server.key -out server.crt -subj "/CN=localhost"

Then visit on your phone:
    https://<your-lan-ip>:8766/pbc_verifier.html

The browser will show a security warning (self-signed cert) — tap
"Advanced" -> "Proceed" to continue.
"""
import ssl
import http.server
import os
from pathlib import Path

PORT = 8766
CERT = Path(__file__).parent / "server.crt"
KEY  = Path(__file__).parent / "server.key"

if not CERT.exists() or not KEY.exists():
    raise SystemExit(
        "Missing local TLS certificate files.\n"
        "Generate them with:\n"
        "  openssl req -x509 -newkey rsa:2048 -nodes -days 365 "
        "-keyout server.key -out server.crt -subj \"/CN=localhost\"\n"
        "These files are ignored by git and must not be committed."
    )

os.chdir(Path(__file__).parent)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args)

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))

httpd = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

print(f"HTTPS server running on port {PORT}")
print(f"  Desktop : https://localhost:{PORT}/pbc_verifier.html")
print(f"  Phone   : https://<your-lan-ip>:{PORT}/pbc_verifier.html")
print(f"  (Accept the self-signed cert warning in your browser)")
httpd.serve_forever()
