
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8501")
    # Wait for the app to render
    page.wait_for_timeout(10000)
    page.screenshot(path="/home/jules/verification/single_model_view_with_holdout.png")
    browser.close()
