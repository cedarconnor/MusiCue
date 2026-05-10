"""Capture v0.2b screenshots: editorial format selected with markers panel."""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/screenshots")
SONG = "a0639697796de0fe79f8f78707da867bfcb45ee7790dcff2ef49d1f0ce88aa94"
ANALYSIS = "bac98e988355"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(f"http://localhost:8765/editor/{SONG}/{ANALYSIS}")
        for _ in range(60):
            dur = page.evaluate(
                """() => {
                  const h = Array.from(document.querySelectorAll('div')).filter(d=>d.shadowRoot);
                  const a = h[0]?.shadowRoot.querySelector('audio');
                  return a && Number.isFinite(a.duration) ? a.duration : 0;
                }"""
            )
            if dur > 0:
                break
            time.sleep(0.5)
        time.sleep(1.5)

        page.get_by_role("button", name="Export ▶").click()
        time.sleep(0.6)

        # Pick FCPXML — reveals the Markers multi-select.
        page.locator("select").nth(0).select_option("fcpxml")
        time.sleep(0.4)
        page.screenshot(path=str(OUT / "export_modal_fcpxml.png"))
        print("wrote export_modal_fcpxml.png")

        # Pick EDL — same Markers panel, different format header.
        page.locator("select").nth(0).select_option("edl")
        time.sleep(0.4)
        page.screenshot(path=str(OUT / "export_modal_edl.png"))
        print("wrote export_modal_edl.png")

        browser.close()


if __name__ == "__main__":
    main()
