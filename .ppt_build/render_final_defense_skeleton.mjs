import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const runtimeModules = "C:/Users/zhuoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const workspaceDir = "C:/Users/zhuoy/Desktop/无/大四上/小学期/LeNet_optimized_verified/FPGA_LeNet_final";
const deckPath = path.join(workspaceDir, "final_ppt", "LeNet_Final_Defense_Skeleton_v2.pptx");
const outputDir = path.join(workspaceDir, ".ppt_build", "rendered_skeleton");
const { FileBlob, PresentationFile } = await import(pathToFileURL(
  path.join(runtimeModules, "@oai/artifact-tool/dist/artifact_tool.mjs"),
).href);

await fs.mkdir(outputDir, { recursive: true });
const presentation = await PresentationFile.importPptx(await FileBlob.load(deckPath));
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(outputDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log(`Rendered ${presentation.slides.items.length} slides to ${outputDir}`);
