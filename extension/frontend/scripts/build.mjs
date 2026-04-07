import { spawnSync } from "node:child_process";

const target = process.argv[2] ?? "chromium";
const validTargets = new Set(["chromium", "firefox"]);

if (!validTargets.has(target)) {
    console.error(
        `Invalid browser target \"${target}\". Expected one of: ${Array.from(validTargets).join(", ")}`,
    );
    process.exit(1);
}

function run(command, args, env = {}) {
    const result = spawnSync(command, args, {
        stdio: "inherit",
        shell: process.platform === "win32",
        env: {
            ...process.env,
            ...env,
        },
    });

    if (result.status !== 0) {
        process.exit(result.status ?? 1);
    }
}

run("npx", ["tsc", "-b"]);

const viteConfigs = [
    "configs/worker.vite.config.ts",
    "configs/content.vite.config.ts",
    "configs/page.vite.config.ts",
    "configs/popup.vite.config.ts",
];

for (const config of viteConfigs) {
    run("npx", ["vite", "build", "-c", config], {
        BROWSER_TARGET: target,
    });
}
