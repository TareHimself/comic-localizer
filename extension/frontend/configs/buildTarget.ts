export type BrowserTarget = "chromium" | "firefox";

const VALID_TARGETS: BrowserTarget[] = ["chromium", "firefox"];

export function getBrowserTarget(): BrowserTarget {
    const value = process.env.BROWSER_TARGET;
    if (value === undefined || value.trim() === "") {
        return "chromium";
    }

    if (!VALID_TARGETS.includes(value as BrowserTarget)) {
        throw new Error(
            `Invalid BROWSER_TARGET="${value}". Expected one of: ${VALID_TARGETS.join(", ")}`,
        );
    }

    return value as BrowserTarget;
}

export function getTargetOutDir(target = getBrowserTarget()): string {
    return `dist/${target}`;
}
