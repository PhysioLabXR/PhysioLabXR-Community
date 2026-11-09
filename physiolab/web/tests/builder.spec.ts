import { test, expect } from "@playwright/test";
test("collaborator creates, configures, runs, stops, exports and reopens a project", async ({
  page,
}) => {
  await page.goto("/");
  const name = `EMG handoff ${Date.now()}`;
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(
    page.getByRole("button", { name: "Run simulation" }),
  ).toBeVisible();
  await page.getByRole("button", { name: /Bandpass filter/ }).click();
  await page.getByLabel("low_hz", { exact: true }).fill("12");
  await page.getByRole("button", { name: "Save parameters" }).click();
  await expect(page.getByRole("status")).toContainText("Parameters saved");
  await page.getByRole("button", { name: "Run simulation" }).click();
  await expect(page.getByTestId("inference")).toHaveText("gesture", {
    timeout: 12000,
  });
  await page.getByRole("button", { name: "Stop run" }).click();
  await expect(
    page.getByRole("cell", { name: "stopped", exact: true }),
  ).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Raw data + results" }).click();
  const download = await downloadPromise;
  const { readFile } = await import("node:fs/promises");
  const recorded = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(recorded.raw.length).toBeGreaterThan(0);
  expect(
    recorded.inferences.some((i: { label: string }) => i.label === "gesture"),
  ).toBe(true);
  expect(recorded.status).toBe("stopped");
  const appDownloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Web app", exact: true }).click();
  const appDownload = await appDownloadPromise;
  expect(await readFile((await appDownload.path())!, "utf8")).toContain(
    "Replay session",
  );
  await page.reload();
  await page.getByRole("button", { name: new RegExp(name) }).click();
  await expect(
    page.getByRole("cell", { name: "stopped", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Device catalog", exact: true })
    .click();
  await expect(page.getByText("Apple Watch", { exact: true })).toBeVisible();
  await expect(page.getByText("Oura Ring", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "test-results/device-catalog.png",
    fullPage: true,
  });
});
