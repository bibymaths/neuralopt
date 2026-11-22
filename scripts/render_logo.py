"""Re-render the logo PNGs and favicons from the SVG sources in docs/assets/images/.

The SVGs are the sources. The wordmark is Inter Display Bold (178 px, letter-spacing -6),
already converted to paths with ``inkscape --export-text-to-path``, so no font is needed
to render them. Requires Inkscape 1.x on PATH. Usage: ``python scripts/render_logo.py``.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

IMAGES = Path(__file__).parents[1] / "docs" / "assets" / "images"
FAVICON = IMAGES / "favicon"
ICON_PNGS = {
    "favicon-16x16.png": 16,
    "favicon-32x32.png": 32,
    "apple-touch-icon.png": 180,
    "android-chrome-192x192.png": 192,
    "android-chrome-512x512.png": 512,
}


def _png(svg: Path, out: Path, width: int, height: int | None = None) -> None:
    size = ["-w", str(width)] + (["-h", str(height)] if height else [])
    subprocess.run(["inkscape", str(svg), "--export-type=png", f"--export-filename={out}",
                    *size], check=True, capture_output=True)


def _ico(pngs: list[Path], out: Path) -> None:
    """Write a PNG-embedded .ico (supported by every current browser)."""
    blobs = [p.read_bytes() for p in pngs]
    sizes = [struct.unpack(">I", b[16:20])[0] for b in blobs]  # PNG IHDR width
    offset = 6 + 16 * len(blobs)
    header = struct.pack("<HHH", 0, 1, len(blobs))
    entries, data = b"", b""
    for size, blob in zip(sizes, blobs, strict=True):
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(blob),
                               offset + len(data))
        data += blob
    out.write_bytes(header + entries + data)


def main() -> None:
    _png(IMAGES / "logo.svg", IMAGES / "logo.png", 1448, 500)
    _png(IMAGES / "logo-dark.svg", IMAGES / "logo-dark.png", 1448, 500)
    for name, size in ICON_PNGS.items():
        _png(IMAGES / "icon.svg", FAVICON / name, size, size)
    _ico([FAVICON / "favicon-16x16.png", FAVICON / "favicon-32x32.png"], FAVICON / "favicon.ico")


if __name__ == "__main__":
    main()
