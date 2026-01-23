import subprocess
import time
import os
import sys
from playwright.sync_api import sync_playwright

def verify_toast():
    # Start HTTP server
    server = subprocess.Popen(
        ["python3", "-m", "http.server", "8000"],
        cwd="verification/test_env",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        time.sleep(2) # Wait for server to start

        with sync_playwright() as p:
            print("Launching browser...")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Capture console logs
            page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
            page.on("pageerror", lambda err: print(f"BROWSER ERROR: {err}"))

            # Go to the test page
            print("Navigating to page...")
            page.goto("http://localhost:8000/index.html")

            # Click the trigger button
            print("Triggering toast...")
            page.click("#trigger-toast")

            # Wait for toast to appear
            print("Waiting for toast...")
            try:
                toast = page.locator(".comfy-toast-success")
                toast.wait_for(state="visible", timeout=5000)

                # Verify text content
                text = toast.text_content()
                print(f"Toast text: {text}")
                if "Parameters saved successfully!" not in text:
                    print("FAIL: Unexpected toast text")
                    sys.exit(1)

                # Take screenshot
                print("Taking screenshot...")
                screenshot_path = os.path.abspath("verification/toast_notification.png")
                page.screenshot(path=screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
                print("SUCCESS: Toast verified.")
            except Exception as e:
                print(f"Failed to find toast: {e}")
                # Take screenshot anyway for debugging
                page.screenshot(path=os.path.abspath("verification/debug_fail.png"))
                sys.exit(1)

            browser.close()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        server.terminate()
        server.wait()

if __name__ == "__main__":
    verify_toast()
