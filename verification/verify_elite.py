from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        path = os.path.abspath("LPMain6.html")
        page.goto(f"file://{path}")

        # 1. Verify PDF.js Script Presence
        pdf_script = page.locator('script[src*="pdf.min.js"]')
        assert pdf_script.count() > 0
        print("PDF.js script found.")

        # 2. Open Menu
        # Find a menu button. The cards have a "CARTE" button.
        # .card-btn.btn-menu
        menu_btn = page.locator('.card-btn.btn-menu').first
        menu_btn.click()

        # Wait for menu modal
        page.wait_for_selector("#menuModal.active", state="visible")
        print("Menu modal opened.")

        # Wait for fallback OR canvas. Since we are in headless/file:// context,
        # PDF.js worker might fail due to CORS or file protocol limitations in some environments,
        # OR it might work. We check if the container is populated.
        # But wait! 'file://' protocol often blocks fetch() needed by PDF.js.
        # We can at least check that the structure is there.

        # Let's check if the fallback button appears (if PDF load fails) OR canvas appears.
        try:
            page.wait_for_selector("canvas", timeout=5000)
            print("Canvas element rendered (PDF.js working).")
        except:
            print("Canvas not found (expected in file:// protocol due to fetch restrictions).")
            # Check fallback
            page.wait_for_selector(".menu-fallback-container.visible", timeout=5000)
            print("Fallback button visible (Graceful failure confirmed).")

        # 3. Close Menu
        close_btn = page.locator('.close-menu-container')
        close_btn.click()
        page.wait_for_selector("#menuModal.active", state="hidden")
        print("Menu closed.")

        # 4. Open Story
        card = page.locator(".card-image").first
        card.click()
        page.wait_for_selector("#storyOverlay.show", state="visible")
        print("Story opened.")

        # Check for Loader presence (it might be hidden quickly if load is fast, or visible if slow)
        # We just want to ensure the element exists in DOM now
        loader = page.locator("#story-premium-loader")
        # It might be created dynamically
        page.wait_for_timeout(500)
        if loader.count() > 0:
             print("Story Premium Loader element exists.")

        page.screenshot(path="verification/elite_verification.png")

        browser.close()

if __name__ == "__main__":
    run()
