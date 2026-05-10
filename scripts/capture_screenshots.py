"""Capture MusiCue UI screenshots via headed Playwright.

Run after `python -m musicue ui --no-open` is up on localhost:8765 and at least
one fully-analyzed song exists. Writes PNGs into docs/screenshots/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8765"
OUT = Path("docs/screenshots")


def wait_for_decode(page, timeout_s: float = 30.0) -> float:
    """Block until the mix WaveSurfer reports a positive duration."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        dur = page.evaluate(
            """
            () => {
              const hosts = Array.from(document.querySelectorAll('div'))
                .filter(d => d.shadowRoot);
              const a = hosts[0]?.shadowRoot.querySelector('audio');
              return a && Number.isFinite(a.duration) ? a.duration : 0;
            }
            """
        )
        if dur and dur > 0:
            return dur
        time.sleep(0.5)
    raise TimeoutError(f"mix audio never decoded within {timeout_s}s")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    songs = []

    with sync_playwright() as p:
        # Headed so the OS-level audio stack is fully present.
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--autoplay-policy=no-user-gesture-required",
                "--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies",
            ],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Pick the first song with v0.1c-grade analysis.
        page.goto(f"{BASE}/api/songs", wait_until="load")
        songs = page.evaluate("() => JSON.parse(document.body.innerText).songs")
        if not songs:
            print("no songs in library", file=sys.stderr)
            return 1

        # Library page screenshot
        page.goto(f"{BASE}/library", wait_until="networkidle")
        time.sleep(1.0)
        page.screenshot(path=str(OUT / "library.png"), full_page=False)
        print(f"wrote {OUT / 'library.png'}")

        # Pick a song with both curves and section_transitions.
        chosen = None
        for s in songs:
            aid = (s.get("analysis_ids") or [None])[0]
            if not aid:
                continue
            page.goto(f"{BASE}/api/songs/{s['id']}/analyses/{aid}", wait_until="load")
            j = page.evaluate("() => JSON.parse(document.body.innerText)")
            tr = (j.get("section_transitions") or [])
            curves = j.get("curves") or {}
            if tr and "lufs" in curves:
                chosen = (s, aid)
                break
        if not chosen:
            print("no song has section_transitions + curves", file=sys.stderr)
            return 1
        s, aid = chosen
        print(f"chosen: {s['title']!r} (analysis {aid})")

        # Editor page
        page.goto(f"{BASE}/editor/{s['id']}/{aid}", wait_until="domcontentloaded")
        dur = wait_for_decode(page, timeout_s=30.0)
        print(f"mix decoded: {dur:.2f}s")
        # Let WaveSurfer paint and overlays draw.
        time.sleep(2.0)

        # Editor — default view (curves expanded, LUFS, fixed range)
        page.screenshot(path=str(OUT / "editor_overview.png"), full_page=False)
        print(f"wrote {OUT / 'editor_overview.png'}")

        # Zoom in for a closer look at section bar + ramps
        zoom_slider = page.locator('input[type="range"]')
        if zoom_slider.count():
            zoom_slider.first.fill("4")
            time.sleep(0.5)
            page.screenshot(path=str(OUT / "editor_zoomed.png"), full_page=False)
            print(f"wrote {OUT / 'editor_zoomed.png'}")
            # Reset zoom
            zoom_slider.first.fill("1")
            time.sleep(0.3)

        # Switch curve to Spectral Centroid
        page.locator("select").first.select_option(value="spectral_centroid")
        time.sleep(0.6)
        page.screenshot(path=str(OUT / "editor_curve_centroid.png"), full_page=False)
        print(f"wrote {OUT / 'editor_curve_centroid.png'}")

        # Toggle autoscale
        page.locator('button:has-text("fixed range")').click()
        time.sleep(0.4)
        page.screenshot(path=str(OUT / "editor_curve_autoscale.png"), full_page=False)
        print(f"wrote {OUT / 'editor_curve_autoscale.png'}")

        # Restore fixed range, switch back to LUFS
        page.locator('button:has-text("autoscale")').click()
        time.sleep(0.2)
        page.locator("select").first.select_option(value="lufs")
        time.sleep(0.3)

        # Click somewhere on the mix lane to seek into a chorus, then play briefly
        # so the cursor shows on the curve canvas.
        mix_host = page.locator("div").filter(has_text="").nth(0)
        page.evaluate(
            """
            () => {
              const hosts = Array.from(document.querySelectorAll('div'))
                .filter(d => d.shadowRoot);
              const a = hosts[0]?.shadowRoot.querySelector('audio');
              if (a) a.currentTime = a.duration * 0.42;
            }
            """
        )
        time.sleep(0.6)
        page.screenshot(path=str(OUT / "editor_cursor_synced.png"), full_page=False)
        print(f"wrote {OUT / 'editor_cursor_synced.png'}")

        # Collapse the curves panel — RMS tint should disappear from stem lanes.
        page.locator('button:has-text("Curves")').click()
        time.sleep(0.4)
        page.screenshot(path=str(OUT / "editor_curves_collapsed.png"), full_page=False)
        print(f"wrote {OUT / 'editor_curves_collapsed.png'}")

        # Expand again for a hover-on-transition shot.
        page.locator('button:has-text("Curves")').click()
        time.sleep(0.3)

        # Hover the FIRST transition rect to show its tooltip.
        # Get its viewport coords from the analysis JSON.
        tooltip_xy = page.evaluate(
            """
            () => {
              const rects = Array.from(document.querySelectorAll('div'))
                .filter(d => d.style.cursor === 'pointer' && d.style.position === 'absolute' && d.parentElement?.style?.position === 'absolute');
              const r = rects[0]?.getBoundingClientRect();
              return r ? { x: r.left + r.width / 2, y: r.top + r.height / 2 } : null;
            }
            """
        )
        if tooltip_xy:
            page.mouse.move(tooltip_xy["x"], tooltip_xy["y"])
            time.sleep(0.6)
            page.screenshot(path=str(OUT / "editor_transition_tooltip.png"), full_page=False)
            print(f"wrote {OUT / 'editor_transition_tooltip.png'}")

        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
