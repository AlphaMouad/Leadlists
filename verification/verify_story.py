import os
from playwright.sync_api import sync_playwright

def verify_story():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Emulate mobile to check responsiveness too
        context = browser.new_context(
            viewport={'width': 390, 'height': 844},
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
        )
        page = context.new_page()

        # Get absolute path
        cwd = os.getcwd()
        file_path = f"file://{cwd}/LPMain1.html"
        print(f"Navigating to {file_path}")

        page.goto(file_path)

        # Wait for card
        print("Waiting for card...")
        page.wait_for_selector("#card-nouba")

        # Click the card image wrapper
        print("Clicking card...")
        page.click("#card-nouba .card-image")

        # Wait for overlay
        print("Waiting for overlay...")
        page.wait_for_selector("#storyOverlay.show")

        # Verify Poster Injection
        print("Waiting for temp-poster...")
        page.wait_for_selector("#temp-poster", state="visible", timeout=5000)

        # Wait a tiny bit to ensure rendering
        page.wait_for_timeout(500)

        # Take screenshot
        output_path = f"{cwd}/verification/story_verification.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    try:
        verify_story()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
