
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:8501")
    # Click the "Model Comparison View" radio button
    page.click('label:has-text("Model Comparison View")')
    # Wait for the app to re-render
    page.wait_for_timeout(10000)
    page.screenshot(path="/home/jules/verification/comparison_view.png")
    browser.close()
