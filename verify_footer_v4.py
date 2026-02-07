
import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        # iPhone 12 Pro emulation
        iphone_12 = p.devices['iPhone 12 Pro']
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(**iphone_12)
        page = await context.new_page()

        # Load the page
        await page.goto('http://localhost:8000/LP1FinalDark99.html')

        # Wait for cards to be generated
        try:
            await page.wait_for_selector('.spectacle-card', state='visible', timeout=10000)
            print("Cards visible.")
        except Exception as e:
            print(f"Error waiting for cards: {e}")
            await page.screenshot(path='error_no_cards.png')
            await browser.close()
            return

        # Click the first card image to open the story
        try:
            # Force click in case of overlay/interception
            await page.click('.spectacle-card:first-child .card-image', force=True)
            print("Clicked card.")
        except Exception as e:
            print(f"Error clicking card: {e}")
            await page.screenshot(path='error_click_v4.png')
            await browser.close()
            return

        # Wait for story overlay
        try:
            await page.wait_for_selector('#storyOverlay.show', state='visible', timeout=5000)
            print("Story overlay visible.")

            # Wait a bit for animations
            await page.wait_for_timeout(2000)

            # Screenshot the story view (focusing on footer)
            await page.screenshot(path='verification_story_footer_v4.png')
            print("Screenshot taken: verification_story_footer_v4.png")

        except Exception as e:
            print(f"Error waiting for story overlay: {e}")
            await page.screenshot(path='error_overlay_v4.png')

        await browser.close()

if __name__ == '__main__':
    asyncio.run(run())
