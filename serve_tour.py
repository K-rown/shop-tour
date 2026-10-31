"""Serve a built tour folder, and let the viewer save its pinned notes: python serve_tour.py <tour folder> [--bind 127.0.0.1] [--port 8000]

Everything is plain static files (GET) except one thing: the viewer's "Notes" save a list to notes.json. When the viewer is opened from this server it sends the
list with PUT /notes.json; the server writes it next to the tour (tour/notes.json, so it is exported with the tour) and in the walk's own folder
(<walk>/notes.json, the master copy that every rebuild copies back into tour/). Nothing else can be written, the body is limited to 8 MB and must be a JSON
list of notes (each with an id, a place and a text or photo). On any other host (GitHub Pages, a file share) the viewer keeps the notes in the browser and offers
an "Export notes" download instead; put that notes.json next to index.html to share it.
"""
import argparse, functools, http.server, json, os, sys, tempfile

MAX = 8 * 1024 * 1024


def valid(notes):
    if not isinstance(notes, list) or len(notes) > 2000:
        return False
    for n in notes:
        if not isinstance(n, dict) or not isinstance(n.get("id"), str) or len(n["id"]) > 64:
            return False
        if not all(isinstance(n.get(k), (int, float)) for k in ("x", "y", "z")):
            return False
        if not isinstance(n.get("text", ""), str) or len(n.get("text", "")) > 4000:
            return False
        ph = n.get("photo")
        if ph is not None and not (isinstance(ph, str) and ph.startswith("data:image/") and len(ph) < 1_500_000):
            return False
    return True


def write_atomic(path, data):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".notes-", suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def handler_for(tour_dir):
    walk_dir = os.path.dirname(os.path.abspath(tour_dir))

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            if os.environ.get("SHOP_TOUR_SERVE_LOG"):
                super().log_message(*a, **k)

        def do_PUT(self):
            if self.path.split("?")[0] != "/notes.json":
                return self.send_error(405, "only /notes.json can be written")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX:
                return self.send_error(413, "too big")
            body = self.rfile.read(n)
            try:
                notes = json.loads(body)
            except ValueError:
                return self.send_error(400, "not JSON")
            if not valid(notes):
                return self.send_error(400, "not a list of notes")
            data = json.dumps(notes, indent=1).encode("utf-8")
            write_atomic(os.path.join(os.path.abspath(tour_dir), "notes.json"), data)
            write_atomic(os.path.join(walk_dir, "notes.json"), data)
            tj = os.path.join(os.path.abspath(tour_dir), "tour.json")                         # tell the viewer there are notes to fetch
            try:
                with open(tj, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if not meta.get("notes"):
                    meta["notes"] = True
                    write_atomic(tj, json.dumps(meta, separators=(",", ":")).encode("utf-8"))
            except (OSError, ValueError):
                pass
            self.send_response(204); self.end_headers()

        do_POST = do_PUT

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

    return functools.partial(H, directory=os.path.abspath(tour_dir))


class Srv(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):           # a browser that closes a connection mid-download is normal
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tour"); ap.add_argument("--bind", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = Srv((a.bind, a.port), handler_for(a.tour))
    print(f"serving {a.tour} at http://{'localhost' if a.bind in ('127.0.0.1', '') else a.bind}:{a.port}  (notes are saved to {os.path.dirname(os.path.abspath(a.tour))}{os.sep}notes.json)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
