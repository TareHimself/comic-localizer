import { defineConfig } from "vite";
import { getBrowserTarget, getTargetOutDir } from "./buildTarget";

// https://vite.dev/config/
const target = getBrowserTarget();

export default defineConfig({
    plugins: [],
    build: {
        minify: false,
        sourcemap: true,
        emptyOutDir: false,
        outDir: getTargetOutDir(target),
        rollupOptions: {
            input: {
                page: "src/page.ts",
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
