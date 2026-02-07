from playwright.sync_api import sync_playwright
import time

def verify_story_footer():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Emulate iPhone 12 Pro for mobile view verification
        context = browser.new_context(
            viewport={'width': 390, 'height': 844},
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
        )
        page = context.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page Error: {err}"))

        # Load the page (assuming local server is running on 8080)
        page.goto('http://localhost:8080/LP1FinalDark99.html')

        print("Page loaded.")

        # Scroll down to trigger any lazy loading or visibility checks
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)

        # Wait for cards to be generated
        try:
            # wait for .card-image specifically as that's what we click
            page.wait_for_selector('.card-image', state='visible', timeout=10000)
            print("Cards visible.")
        except Exception as e:
            print(f"Error waiting for cards: {e}")
            page.screenshot(path='error_no_cards.png')
            browser.close()
            return

        # Click the first card's image to open the story
        try:
            # Use specific selector to avoid ambiguity
            page.click('.card:first-child .card-image')
            print("Clicked card.")
        except Exception as e:
            print(f"Error clicking card: {e}")
            page.screenshot(path='error_click.png')
            browser.close()
            return

        # Wait for the story overlay to be visible
        try:
            page.wait_for_selector('.story-overlay', state='visible', timeout=10000)
            print("Story overlay visible.")
        except Exception as e:
            print(f"Error waiting for overlay: {e}")
            page.screenshot(path='error_no_overlay.png')
            browser.close()
            return

        # Wait a bit for animations
        time.sleep(3)

        # Take a screenshot
        page.screenshot(path='verification_story_v3.png')
        print("Story screenshot v3 taken.")

        browser.close()

if __name__ == "__main__":
    verify_story_footer()
