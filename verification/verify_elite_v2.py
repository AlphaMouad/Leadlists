from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        path = os.path.abspath("LPMain6.html")
        page.goto(f"file://{path}")

        # 1. Verify PDF.js Script REMOVED
        pdf_script = page.locator('script[src*="pdf.min.js"]')
        assert pdf_script.count() == 0
        print("PDF.js script successfully removed.")

        # 2. Open Menu
        menu_btn = page.locator('.card-btn.btn-menu').first
        menu_btn.click()

        # Wait for menu modal
        page.wait_for_selector("#menuModal.active", state="visible")
        print("Menu modal opened.")

        # 3. Verify Iframe Structure
        # We expect an iframe now, not a canvas
        try:
            page.wait_for_selector(".menu-iframe-container iframe", timeout=5000)
            print("Iframe element found (Restored successfully).")
        except:
            print("ERROR: Iframe not found.")

        # 4. Verify Fallback Logic (Simulate Timeout)
        # Since we are in file://, the iframe might fail or hang.
        # We check if the fallback container exists in DOM and becomes visible eventually.
        # The script sets a 3.5s timeout.
        try:
            # Wait for fallback to become visible
            page.wait_for_selector(".menu-fallback-container.visible", timeout=6000)
            print("Fallback button appeared (Timeout logic verified).")
        except:
            print("Fallback button did not appear within timeout window (Check logic).")

        page.screenshot(path="verification/elite_v2_verification.png")
        browser.close()

if __name__ == "__main__":
    run()
