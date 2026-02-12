from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8080/LPMain4.html")

        # Wait for content
        page.wait_for_selector(".spectacle-card")

        # Click the Nouba card (has video)
        page.click(".spectacle-card[data-id='nouba'] .card-image")

        # Wait for overlay to appear
        page.wait_for_selector("#storyOverlay.show")

        # Wait a bit
        time.sleep(3)

        # Take screenshot
        page.screenshot(path="verification/story_open.png")

        browser.close()

if __name__ == "__main__":
    run()
