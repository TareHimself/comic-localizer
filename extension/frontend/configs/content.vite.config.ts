import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import { getBrowserTarget, getTargetOutDir } from "./buildTarget";

// https://vite.dev/config/
const target = getBrowserTarget();

export default defineConfig({
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: "public/page_styles.css",
                    dest: ".",
                },
            ],
        }),
    ],
    build: {
        minify: false,
        sourcemap: true,
        emptyOutDir: false,
        outDir: getTargetOutDir(target),
        rollupOptions: {
            input: {
                content: "src/content/content.ts",
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
