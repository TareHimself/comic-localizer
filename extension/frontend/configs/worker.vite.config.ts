import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { getBrowserTarget, getTargetOutDir } from "./buildTarget";

// https://vite.dev/config/
const target = getBrowserTarget();

export default defineConfig({
    plugins: [
        react({
            babel: {
                plugins: [["babel-plugin-react-compiler"]],
            },
        }),
    ],
    build: {
        minify: false,
        sourcemap: true,
        emptyOutDir: true,
        outDir: getTargetOutDir(target),
        rollupOptions: {
            input: {
                worker: "src/worker/worker.ts",
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
