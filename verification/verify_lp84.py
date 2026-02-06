from playwright.sync_api import sync_playwright, expect
import re
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 375, 'height': 812})

        page.goto("http://localhost:8000/LP1FinalDark84.html")

        # Verify nouba sold out
        nouba_card = page.locator("#card-nouba")
        expect(nouba_card).to_have_class(re.compile(r"sold-out"), timeout=10000)
        expect(nouba_card.locator(".sold-out-badge")).to_have_text("COMPLET CE SOIR")
        page.screenshot(path="verification/nouba_sold_out.png")

        # Verify Phone Validation
        print("Verifying phone validation...")
        page.locator("#fullname").fill("Test User")
        page.locator("#date").fill("2025-12-31")
        page.select_option("#restaurant", "folk")
        page.select_option("#guests", "2")

        page.locator("#phone").fill("1234")
        page.click("#submitBtn", force=True)

        error_msg = page.locator("#error-phone")
        expect(error_msg).to_be_visible()
        expect(error_msg).to_contain_text("Numéro mobile invalide")

        print("Taking screenshot of form error...")
        page.screenshot(path="verification/phone_error.png")

        browser.close()

if __name__ == "__main__":
    run()
