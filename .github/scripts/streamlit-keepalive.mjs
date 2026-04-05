import { chromium } from "playwright";

const appUrl = (process.env.STREAMLIT_APP_URL || "").trim();

if (!appUrl) {
  throw new Error("Missing STREAMLIT_APP_URL.");
}

const sleepingPatterns = [
  /gone to sleep/i,
  /get this app back up/i,
  /is sleeping/i,
  /wake (it|this app) up/i,
];

const wakeButtonSelectors = [
  'button:has-text("Yes, get this app back up!")',
  'button:has-text("Get this app back up")',
  'button:has-text("Wake this app up")',
  'button:has-text("Wake it up")',
  'button:has-text("Wake")',
  'text=/get this app back up/i',
  'text=/wake .*app/i',
];

function looksSleeping(text) {
  return sleepingPatterns.some((pattern) => pattern.test(text));
}

async function bodyText(page) {
  const text = await page.locator("body").innerText().catch(() => "");
  return String(text || "").trim();
}

async function clickWakeButton(page) {
  for (const selector of wakeButtonSelectors) {
    const locator = page.locator(selector).first();
    const count = await locator.count().catch(() => 0);

    if (!count) {
      continue;
    }

    await locator.click({ timeout: 5000 });
    return true;
  }

  return false;
}

async function waitForHealthyApp(page) {
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    await page.waitForLoadState("domcontentloaded", { timeout: 120000 }).catch(() => {});
    await page.waitForTimeout(5000);

    const currentText = await bodyText(page);
    console.log(`Attempt ${attempt}: ${currentText.slice(0, 160).replace(/\s+/g, " ")}`);

    if (/sign in|log in/i.test(currentText)) {
      throw new Error("The Streamlit app appears to require sign-in. This workflow expects a public app.");
    }

    if (!looksSleeping(currentText)) {
      await page
        .waitForSelector('[data-testid="stAppViewContainer"], .stApp, section.main', {
          timeout: 20000,
        })
        .catch(() => {});

      const healthyText = await bodyText(page);
      if (!looksSleeping(healthyText)) {
        return;
      }
    }

    const clicked = await clickWakeButton(page);
    if (clicked) {
      console.log("Wake button found. Waiting for the app to come back.");
      await page.waitForTimeout(15000);
    } else {
      console.log("Wake button not found. Reloading the page.");
      await page.reload({ waitUntil: "domcontentloaded", timeout: 120000 }).catch(() => {});
      await page.waitForTimeout(10000);
    }
  }

  const finalText = await bodyText(page);
  throw new Error(
    `The app still looks asleep or unhealthy after retries. Body snippet: ${finalText
      .slice(0, 250)
      .replace(/\s+/g, " ")}`
  );
}

const browser = await chromium.launch({
  headless: true,
  args: ["--disable-dev-shm-usage"],
});

try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1024 },
  });

  page.setDefaultTimeout(30000);
  await page.goto(appUrl, { waitUntil: "domcontentloaded", timeout: 120000 });
  await waitForHealthyApp(page);
  console.log("Streamlit app is reachable.");
} finally {
  await browser.close();
}
