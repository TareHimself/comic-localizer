import fs from "node:fs";
import path from "node:path";
import type { BrowserTarget } from "./buildTarget";

type ExtensionManifest = Record<string, unknown>;

const BASE_MANIFEST_PATH = path.resolve(process.cwd(), "public/manifest.json");

export function loadBaseManifest(): ExtensionManifest {
    const content = fs.readFileSync(BASE_MANIFEST_PATH, "utf-8");
    return JSON.parse(content) as ExtensionManifest;
}

export function buildManifest(target: BrowserTarget): ExtensionManifest {
    const base = loadBaseManifest();
    if (target !== "firefox") {
        return base;
    }

    return {
        ...base,
        browser_specific_settings: {
            gecko: {
                id: "manga-translator@oyintare.dev",
            },
        },
    };
}
