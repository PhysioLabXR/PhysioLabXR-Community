import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:5178", trace: "retain-on-failure" },
  webServer: [
    {
      command:
        "uv run --directory ../backend uvicorn physiolab.api:app --host 127.0.0.1 --port 8008",
      url: "http://127.0.0.1:8008/api/health",
      reuseExistingServer: !process.env.CI,
      env: { PHYSIOLAB_DATA_DIR: "../.data/e2e" },
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5178",
      reuseExistingServer: !process.env.CI,
    },
  ],
});
