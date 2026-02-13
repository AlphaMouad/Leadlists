from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load local file
        path = os.path.abspath("LPMain6.html")
        page.goto(f"file://{path}")

        # 1. Verify Metadata
        title = page.title()
        desc = page.locator('meta[name="description"]').get_attribute("content")

        print(f"Title: {title}")
        print(f"Description: {desc}")

        assert "Top Spot Elite" in title
        assert "Découvrez les 7 meilleurs" in desc

        # 2. Screenshot Initial State
        page.screenshot(path="verification/lp6_initial.png")

        # 3. Open a Story (Click on first card)
        # Find a card image
        card = page.locator(".card-image").first
        card.click()

        # Wait for overlay to show
        page.wait_for_selector("#storyOverlay.show", state="visible")

        # Screenshot Story Mode
        # Wait a bit for potential video load logic to trigger (though headless might block autoplay)
        page.wait_for_timeout(1000)
        page.screenshot(path="verification/lp6_story.png")

        browser.close()

if __name__ == "__main__":
    run()
