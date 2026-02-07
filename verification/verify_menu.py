from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    # iPhone 12 Pro viewport
    context = browser.new_context(
        viewport={'width': 390, 'height': 844},
        user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1'
    )
    page = context.new_page()

    print("Navigating to page...")
    page.goto("http://localhost:8000/LP1FinalDark119.html")

    print("Waiting for cards...")
    # Wait for cards to appear (reveal animation)
    page.wait_for_selector(".spectacle-card", state="visible", timeout=10000)

    print("Clicking Menu button...")
    # Find the first MENU button
    # The button text is "MENU" inside a button with class "btn-menu"
    menu_btns = page.query_selector_all(".btn-menu")
    if len(menu_btns) > 0:
        menu_btns[0].click()
    else:
        print("No menu buttons found!")
        browser.close()
        return

    print("Waiting for modal...")
    page.wait_for_selector("#menuModal.active", state="visible", timeout=5000)

    print("Waiting for loader/content...")
    # Wait a bit to see what happens (loader or fallback)
    page.wait_for_timeout(3000)

    print("Taking screenshot...")
    page.screenshot(path="verification/menu_modal.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
