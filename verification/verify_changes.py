from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 375, "height": 812}) # Mobile viewport

        # 1. Load Page
        page.goto("http://localhost:8080/LP1FinalDark101.html")
        page.wait_for_load_state("networkidle")

        # 3. Open Story
        page.evaluate("StoryManager.open('malak')")
        page.wait_for_selector(".story-overlay.show")
        time.sleep(2) # Wait for animations

        # 4. Screenshot Story Footer
        story_footer = page.locator(".story-footer")
        # To see the full width effect, we should probably screenshot the bottom of the viewport
        # But locator screenshot is good to see the element boundaries (or lack thereof)
        # Let's take a viewport screenshot cropped to bottom
        page.screenshot(path="verification/story_footer_context.png", clip={"x": 0, "y": 500, "width": 375, "height": 312})

        print("Story footer screenshot taken.")

        browser.close()

if __name__ == "__main__":
    run()
