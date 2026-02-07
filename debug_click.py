
from playwright.sync_api import sync_playwright
import http.server
import socketserver
import threading
import os
import time

PORT = 8003
FILE_TO_SERVE = "LP1FinalDark99.html"

def serve_file():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

def run_debug():
    # Start server in background
    thread = threading.Thread(target=serve_file, daemon=True)
    thread.start()
    time.sleep(1)  # Wait for server to start

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console logs
        page.on("console", lambda msg: print(f"PAGE CONSOLE: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))

        try:
            print(f"Navigating to http://localhost:{PORT}/{FILE_TO_SERVE}")
            page.goto(f"http://localhost:{PORT}/{FILE_TO_SERVE}")
            page.wait_for_load_state("networkidle")

            # Check if StoryManager exists
            is_defined = page.evaluate("typeof StoryManager !== 'undefined'")
            print(f"StoryManager defined: {is_defined}")

            # Check if card exists
            card_count = page.locator(".spectacle-card").count()
            print(f"Card count: {card_count}")

            if card_count > 0:
                # Try to trigger the click
                print("Attempting to click first card image...")
                card_image = page.locator(".spectacle-card").first.locator(".card-image")

                # Highlight element
                card_image.evaluate("el => el.style.border = '5px solid red'")
                page.screenshot(path="debug_before_click.png")

                # Click
                card_image.click(force=True)

                # Wait for overlay
                try:
                    page.wait_for_selector("#storyOverlay.show", timeout=5000)
                    print("SUCCESS: Overlay opened.")
                except Exception as e:
                    print(f"FAILURE: Overlay did not open. Error: {e}")
                    page.screenshot(path="debug_after_fail.png")

                    # Try manual JS call
                    print("Attempting manual JS call to StoryManager.open('malak')...")
                    page.evaluate("StoryManager.open('malak')")
                    try:
                        page.wait_for_selector("#storyOverlay.show", timeout=5000)
                        print("SUCCESS: Overlay opened via JS.")
                    except:
                        print("FAILURE: Overlay did not open via JS either.")

        except Exception as e:
            print(f"An error occurred: {e}")

        browser.close()

if __name__ == "__main__":
    run_debug()
