import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";
import fs from "node:fs";
import path from "node:path";
import { buildManifest } from "./manifest";
import { getBrowserTarget, getTargetOutDir } from "./buildTarget";

// https://vite.dev/config/
const target = getBrowserTarget();
const outDir = getTargetOutDir(target);

export default defineConfig({
    plugins: [
        react({
            babel: {
                plugins: [["babel-plugin-react-compiler"]],
            },
        }),
        viteStaticCopy({
            targets: [
                {
                    src: "public/extension_icon128.png",
                    dest: ".",
                },
            ],
        }),
        {
            name: "generate-browser-manifest",
            apply: "build",
            closeBundle() {
                const manifest = buildManifest(target);
                const outputPath = path.resolve(process.cwd(), outDir, "manifest.json");
                fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 4));
            },
        },
    ],
    build: {
        minify: false,
        sourcemap: true,
        emptyOutDir: false,
        outDir,
        rollupOptions: {
            input: {
                popup: "./popup.html",
            },
            output: {
                entryFileNames: "[name].js",
                chunkFileNames: `assets/[name].js`,
                assetFileNames: "[name][extname]",
                inlineDynamicImports: true,
            },
        },
    },
});
