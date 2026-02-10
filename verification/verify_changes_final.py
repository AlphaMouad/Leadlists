
from playwright.sync_api import sync_playwright
import time
import http.server
import socketserver
import threading
import os

PORT = 8011

def start_server():
    os.chdir('.')
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

def verify():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # Wait for server to start

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 390, 'height': 844}) # Mobile viewport
        page.goto(f'http://localhost:{PORT}/LP1FinalDark101.html')

        # 1. Verify Hero Trust Text (Static)
        # Check if any animation class or style is present on .form-trust-text
        # We can check computed style for 'animation-name'
        hero_trust_anim = page.evaluate("window.getComputedStyle(document.querySelector('.form-trust-text')).animationName")
        print(f"Hero Trust Text Animation: {hero_trust_anim}") # Should be 'none'

        page.screenshot(path='verification/hero_trust_static.png', clip={'x': 0, 'y': 0, 'width': 390, 'height': 400})

        # 2. Verify Footer Trust Badge Removed
        # Open a story to see the footer
        page.evaluate("StoryManager.open('malak')")
        time.sleep(1) # Wait for overlay

        # Take screenshot of the footer area
        # The footer is absolute bottom
        footer = page.locator('.story-footer')
        if footer.is_visible():
            footer.screenshot(path='verification/story_footer_removed_badge.png')
        else:
            print("Footer not visible!")

        # 3. Verify Climax Card Alignment
        # Force show climax card
        page.evaluate("StoryManager.showClimax()")
        time.sleep(1) # Wait for transition

        # Take screenshot of climax card
        climax = page.locator('#climaxCard')
        climax.screenshot(path='verification/climax_card_aligned.png')

        browser.close()

if __name__ == "__main__":
    verify()
