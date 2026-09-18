import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspaceDir = "C:/Users/zhuoy/Desktop/无/大四上/小学期/LeNet_optimized_verified/FPGA_LeNet_final";
const skillDir = "C:/Users/zhuoy/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const runtimeModules = "C:/Users/zhuoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const buildDir = path.join(workspaceDir, ".ppt_build");
const finalDir = path.join(workspaceDir, "final_ppt");
const candidatePath = path.join(buildDir, "LeNet_Final_Defense_Skeleton_v2.candidate.pptx");
const finalPath = path.join(finalDir, "LeNet_Final_Defense_Skeleton_v2.pptx");

const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(runtimeModules, "@oai/artifact-tool/dist/artifact_tool.mjs"),
).href);
const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(
  path.join(skillDir, "container_tools/artifact_tool_utils.mjs"),
).href);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(finalDir, { recursive: true });
const font = resolvePresentationFont({ fontFamily: "Microsoft YaHei" });
const palette = { navy: "#12263F", blue: "#1D5D8F", cyan: "#2C9AB7", ink: "#17212B", muted: "#52616B", paper: "#F7F9FB", rule: "#D7E1E8" };
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function box(slide, text, left, top, width, height, size, color = palette.ink, options = {}) {
  const shape = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: size, color, bold: options.bold ?? false, autoFit: "shrinkText", verticalAlignment: options.verticalAlignment ?? "top", paragraphSpacing: 8 };
  return shape;
}

function band(slide, section, page) {
  slide.background.fill = palette.paper;
  const bar = slide.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 1280, height: 16 }, fill: palette.blue, line: { fill: palette.blue, width: 0 } });
  box(slide, section, 72, 665, 900, 22, 13, palette.muted);
  box(slide, String(page).padStart(2, "0"), 1142, 665, 64, 22, 13, palette.muted, { bold: true });
}

function title(slide, value, subtitle, page, section = "LeNet 推理加速器") {
  band(slide, section, page);
  box(slide, value, 72, 55, 1136, 58, 34, palette.navy, { bold: true });
  if (subtitle) box(slide, subtitle, 72, 121, 1100, 30, 17, palette.muted);
}

function line(slide, left, top, width, color = palette.rule) {
  slide.shapes.add({ geometry: "rect", position: { left, top, width, height: 2 }, fill: color, line: { fill: color, width: 0 } });
}

function keyValue(slide, x, y, label, value, note = "") {
  box(slide, label, x, y, 260, 22, 16, palette.muted);
  box(slide, value, x, y + 27, 300, 68, 26, palette.navy, { bold: true });
  if (note) box(slide, note, x, y + 91, 310, 38, 14, palette.muted);
}

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = palette.navy;
  slide.shapes.add({ geometry: "rect", position: { left: 72, top: 115, width: 10, height: 338 }, fill: palette.cyan, line: { fill: palette.cyan, width: 0 } });
  box(slide, "LeNet 推理加速器", 116, 132, 920, 74, 46, "#FFFFFF", { bold: true });
  box(slide, "Level 1 优化验证与 Level 2 自采数据验证", 120, 221, 860, 40, 24, "#DCEAF4");
  line(slide, 120, 294, 460, "#5A7994");
  box(slide, "智能芯片与系统设计综合实践", 120, 323, 760, 28, 18, "#DCEAF4");
  box(slide, "最终答辩合并骨架\n成员 1 负责统一与合并", 120, 404, 610, 62, 18, "#FFFFFF");
  box(slide, "2026 年 9 月", 120, 590, 300, 22, 15, "#AFC8D9");
  slide.speakerNotes.textFrame.setText("骨架页。最终由成员 1 合并成员 3 至成员 8 的已完成页面。");
}

// 2. Scope
{
  const slide = presentation.slides.add();
  title(slide, "项目范围与交付结构", "主线使用 optimized，高速版本 aggressive5b 仅作精度与延迟权衡对照", 2);
  line(slide, 72, 180, 1136);
  keyValue(slide, 72, 220, "网络", "LeNet，1×28×28 输入", "无 Bias；CHW、OIHW 与 FC [OUT][IN] 已冻结");
  keyValue(slide, 425, 220, "目标器件", "xc7z020clg400-1", "10 ns 约束；Vivado HLS 2018.3");
  keyValue(slide, 778, 220, "定点格式", "W12/A12", "A12：signed I7/F5，scale 为 1/32");
  box(slide, "Level 1", 72, 405, 210, 28, 20, palette.blue, { bold: true });
  box(slide, "MNIST 功能、定点一致性、HLS 优化与 RTL Co-sim", 72, 445, 470, 55, 19);
  box(slide, "Level 2", 650, 405, 210, 28, 20, palette.blue, { bold: true });
  box(slide, "真实手写样本、HLS C/C++ 预处理、Fixed/HLS 批量推理与误差分析", 650, 445, 500, 55, 19);
  slide.speakerNotes.textFrame.setText("来源：README.md、config/quant_config.json 与 Level 2 分工计划。");
}

// 3. Baseline
{
  const slide = presentation.slides.add();
  title(slide, "Level 1 冻结基线", "真实单图 latency 采用 RTL Co-sim，不采用 HLS 状态机保守上界", 3, "Level 1 基础结果");
  keyValue(slide, 72, 210, "Float MNIST Accuracy", "98.97%", "Python Float Baseline");
  keyValue(slide, 420, 210, "Fixed / optimized Accuracy", "98.94%", "10,000 张 HLS CSim");
  keyValue(slide, 768, 210, "baseline RTL latency", "64,815 cycles", "单图真实 RTL Co-sim");
  line(slide, 72, 375, 1136);
  box(slide, "验证口径", 72, 420, 230, 30, 22, palette.blue, { bold: true });
  box(slide, "Float、Fixed 与 HLS 输入输出布局对齐；optimized 的 100,000 个最终 raw logits 与 Python Fixed 一致。", 72, 468, 1040, 54, 21);
  box(slide, "成员 6 页面需要解释：474,826 cycles 是原 Controller 的保守上界，不能写成单图真实 latency。", 72, 555, 1040, 32, 16, "#8B3A3A");
  slide.speakerNotes.textFrame.setText("来源：optimization/REPORT_CN.md、Level 2 分工计划。数值不应由其他成员修改。");
}

// 4. Optimized
{
  const slide = presentation.slides.add();
  title(slide, "optimized 高精度工作点", "Memory banking、融合 ReLU/Pool、消除 Flatten 复制与 FC 并行共同降低延迟", 4, "高精度优化");
  keyValue(slide, 72, 210, "RTL latency", "64,815 → 18,428 cycles", "3.52×；Accuracy 保持 98.94%");
  keyValue(slide, 470, 210, "Conv1", "33,913 → 7,849 cycles", "成员 4：访存银行与 II 解释");
  keyValue(slide, 850, 210, "Conv2", "14,777 → 2,553 cycles", "成员 4：访存银行与 II 解释");
  line(slide, 72, 382, 1136);
  box(slide, "待插入成员 4、5、6 的成品页：问题、修改、前后实测数据与工程结论。", 72, 438, 1030, 32, 22, palette.blue, { bold: true });
  box(slide, "合并时保留各模块的“原来 / 问题 / 修改 / 结果 / 结论”证据链，并统一术语为 baseline、optimized、aggressive5b。", 72, 493, 1045, 62, 19);
  slide.speakerNotes.textFrame.setText("来源：Level 2 分工计划。此页为成员 1 的合并引导页，不替代成员 4 至成员 6 的内容页。");
}

// 5. Tradeoff
{
  const slide = presentation.slides.add();
  title(slide, "精度与延迟工作点", "aggressive5b 改变了通道数，因此是独立工作点，不能与 optimized 混用 logits 或网络定义", 5, "设计权衡");
  keyValue(slide, 72, 225, "optimized", "18,428 cycles", "98.94%，主验证版本");
  keyValue(slide, 455, 225, "aggressive5b", "11,025 cycles", "96.55%，低延迟补充版本");
  keyValue(slide, 838, 225, "aggressive5b 网络变化", "Conv1 6→5；Conv2 16→12", "FC1 使用 24 路并行");
  line(slide, 72, 402, 1136);
  box(slide, "成员 3 的位宽与资源证据页在此处合并。结论必须由 Accuracy、LUT、FF、DSP、BRAM、Latency 和 Clock 的实测数据支撑。", 72, 456, 1050, 62, 20);
  slide.speakerNotes.textFrame.setText("来源：optimization/README.md 与 Level 2 分工计划。");
}

// 6. Level 2 preprocessing
{
  const slide = presentation.slides.add();
  title(slide, "Level 2 HLS C/C++ 预处理", "成员 1 交付。JPEG/PNG 解码仅作输入适配，灰度、ROI、缩放、居中与 A12 导出均由 HLS C++ 核执行", 6, "Level 2 数据通路");
  box(slide, "Simple", 72, 205, 220, 30, 24, palette.blue, { bold: true });
  box(slide, "RGB → 1:2:1 整数灰度 → 28×28", 72, 252, 455, 42, 21);
  box(slide, "Full", 650, 205, 220, 30, 24, palette.blue, { bold: true });
  box(slide, "RGB → 灰度 → 阈值与极性 → ROI → 等比缩放 → 居中 28×28", 650, 252, 490, 58, 21);
  line(slide, 72, 365, 1136);
  keyValue(slide, 72, 410, "当前样本", "55 张", "50 张数字样本，5 张 background 抗干扰样本");
  keyValue(slide, 435, 410, "图像输出", "110 张 PGM", "Simple / Full 各 55 张，28×28 单通道");
  keyValue(slide, 800, 410, "Fixed 输入", "110 份 raw 文件", "每份 784 行，A12 值范围 0 至 32");
  slide.speakerNotes.textFrame.setText("来源：hls/preprocess/level2_preprocess.cpp、data/self_collected/level2_manifest.csv、results/level2_preprocess/README.md。");
}

// 7. HLS verification
{
  const slide = presentation.slides.add();
  title(slide, "预处理核验证与成员 7 交接", "HLS CSim 已通过，Full 核已完成 C synthesis 并生成 RTL", 7, "Level 2 验证准备");
  keyValue(slide, 72, 215, "HLS CSim", "0 errors", "synthetic digit、反色 digit、纯背景与 A12 规则测试");
  keyValue(slide, 445, 215, "C synthesis", "RTL 已生成", "Full 核，xc7z020clg400-1");
  keyValue(slide, 815, 215, "成员 7 输入", "manifest 驱动", "路径、标签、ROI 审计字段均已交接");
  line(slide, 72, 400, 1136);
  box(slide, "0–9 标签进入十分类 Accuracy、每类 Accuracy 与混淆矩阵。background 只报告 prediction、logits、最大 logit 与 Full ROI 状态。", 72, 455, 1040, 62, 21);
  slide.speakerNotes.textFrame.setText("来源：results/level2_preprocess、data/self_collected/HANDOFF_TO_MEMBER7.md。");
}

// 8. Result placeholder
{
  const slide = presentation.slides.add();
  title(slide, "Level 2 批量推理结果", "等待成员 7 交付 Simple / Full 的 Python、Fixed 和 optimized HLS 测试结果", 8, "Level 2 结果");
  box(slide, "成员 7 需提供：总体 Accuracy、每类 Accuracy、Prediction CSV、Mismatch 数量与 logits。", 72, 210, 1030, 34, 22, palette.blue, { bold: true });
  box(slide, "成员 8 需提供：数据集构成、原始样本与预处理示例、混淆矩阵、典型误判和原因分析。", 72, 285, 1030, 34, 22, palette.blue, { bold: true });
  line(slide, 72, 375, 1136);
  box(slide, "合并规则：MNIST Accuracy 与自采数字 Accuracy 分开呈现；background 结果单列为抗干扰观察，不能计入 0–9 Accuracy。", 72, 430, 1040, 56, 21);
  slide.speakerNotes.textFrame.setText("此页保留给成员 7 与成员 8 的成品结果页。禁止在结果到达前填入推测数值。");
}

// 9. Integration gates
{
  const slide = presentation.slides.add();
  title(slide, "合并检查点", "成员 1 统一版式和口径；各成员保留自己负责证据的可追溯来源", 9, "最终答辩整合");
  box(slide, "待收件", 72, 205, 180, 28, 22, palette.blue, { bold: true });
  box(slide, "成员 3、4、5、6、7 的模块 PPT，以及成员 8 的完整 Level 2 PPT。", 72, 250, 1030, 34, 21);
  box(slide, "统一口径", 72, 345, 180, 28, 22, palette.blue, { bold: true });
  box(slide, "baseline / optimized / aggressive5b 名称，Accuracy、Latency、资源数字和真实 RTL latency 解释。", 72, 390, 1030, 34, 21);
  box(slide, "冻结条件", 72, 485, 180, 28, 22, palette.blue, { bold: true });
  box(slide, "Level 2 数据、预处理代码、批测结果、成员页和总 PPT 使用同一份最终清单。", 72, 530, 1030, 34, 21);
  slide.speakerNotes.textFrame.setText("来源：final_ppt/PPT_INTAKE.md 与 Level 2 分工计划。");
}

// 10. Closing
{
  const slide = presentation.slides.add();
  title(slide, "当前结论与待完成事项", "此页在收齐成员结果后再压缩为最终结论页", 10, "最终结论");
  box(slide, "高准确率推荐", 72, 208, 280, 28, 22, palette.blue, { bold: true });
  box(slide, "optimized：98.94%，18,428 cycles", 72, 251, 440, 40, 26, palette.navy, { bold: true });
  box(slide, "低延迟补充工作点", 650, 208, 340, 28, 22, palette.blue, { bold: true });
  box(slide, "aggressive5b：96.55%，11,025 cycles", 650, 251, 470, 40, 26, palette.navy, { bold: true });
  line(slide, 72, 365, 1136);
  box(slide, "下一次合并前必须补齐：成员 7 的 Level 2 最终结果、成员 8 的结果分析页、成员 3–6 的已冻结模块 PPT。", 72, 425, 1050, 62, 21);
  slide.speakerNotes.textFrame.setText("结论中不能把 background 负样本纳入十分类 Accuracy，也不能混用 optimized 与 aggressive5b 的网络定义。");
}

await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:/Users/zhuoy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit", "--validate-heading-punctuation"],
  fontPolicy: { basis: "design", families: [font], scriptFonts: { ea: "Microsoft YaHei" } },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "LeNet_Final_Defense_Skeleton_v2.validation.json"),
};
const result = await finalizePresentation(requirements);
console.log(JSON.stringify({ finalPath: result.finalPath, receiptPath: result.receiptPath, sha256: result.finalSha256 }, null, 2));
