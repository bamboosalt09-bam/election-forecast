import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "..");
const outputDir = path.join(repoRoot, "outputs", "15th_assembly_conversion", "issue_phrase_extraction");
const outputPath = path.join(outputDir, "제15대_국회_어구_이슈_추출_결과.xlsx");

const csvSheets = [
  ["RunSummary", "15th_assembly_run_summary.csv"],
  ["IssueMatches", "15th_assembly_issue_phrase_matches.csv"],
  ["PeriodSummary", "15th_assembly_issue_period_summary.csv"],
  ["SpeakerSummary", "15th_assembly_speaker_issue_summary.csv"],
  ["TermSummary", "15th_assembly_term_summary.csv"],
  ["QualitySummary", "15th_assembly_quality_summary.csv"],
  ["SourceSummary", "15th_assembly_source_issue_summary.csv"],
];

function applyBasicStyle(sheet) {
  const used = sheet.getUsedRange(true);
  if (!used) return;
  used.format.font.name = "맑은 고딕";
  used.format.font.size = 10;
  used.format.verticalAlignment = "top";
  used.format.wrapText = true;

  const header = sheet.getRangeByIndexes(0, 0, 1, used.columnCount);
  header.format.fill.color = "#1F4E78";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  header.format.horizontalAlignment = "center";
  header.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sheet.freezePanes.freezeRows(1);

  for (let col = 0; col < used.columnCount; col += 1) {
    const width = sheet.name === "IssueMatches" && col <= 1 ? 34 : 18;
    sheet.getRangeByIndexes(0, col, Math.max(used.rowCount, 1), 1).format.columnWidth = width;
  }
}

function stripBom(text) {
  return text.replace(/^\uFEFF/, "");
}

const firstCsv = stripBom(await fs.readFile(path.join(outputDir, csvSheets[0][1]), "utf8"));
const workbook = await Workbook.fromCSV(firstCsv, { sheetName: csvSheets[0][0] });
for (const [sheetName, fileName] of csvSheets.slice(1)) {
  const csvText = stripBom(await fs.readFile(path.join(outputDir, fileName), "utf8"));
  await workbook.fromCSV(csvText, { sheetName });
}

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
const runSummaryCsv = stripBom(await fs.readFile(path.join(outputDir, "15th_assembly_run_summary.csv"), "utf8"));
const runMetrics = Object.fromEntries(
  runSummaryCsv
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => {
      const [metric, value] = line.split(",");
      return [metric, value];
    }),
);
const readmeRows = [
  ["항목", "값"],
  ["생성 목적", "제15대 국회 통합 회의록에 대한 issue keyword/phrase matcher 전체 처리 결과"],
  ["입력 행 수", runMetrics.input_speech_rows ?? ""],
  ["매칭된 발언 행 수", runMetrics.matched_speeches ?? ""],
  ["이슈-발언 매칭 행 수", runMetrics.match_issue_rows ?? ""],
  ["고유 이슈 수", runMetrics.unique_issues ?? ""],
  ["고유 어구 수", runMetrics.unique_terms ?? ""],
  ["주의", "PDF 추출 상태가 garbled인 행은 OCR 또는 수동 확인 후 재처리하는 것이 좋습니다."],
];
readme.getRangeByIndexes(0, 0, readmeRows.length, 2).values = readmeRows;
readme.getRange("A1:B1").format.fill.color = "#1F4E78";
readme.getRange("A1:B1").format.font.color = "#FFFFFF";
readme.getRange("A1:B1").format.font.bold = true;
readme.getRange("A:B").format.font.name = "맑은 고딕";
readme.getRange("A:B").format.wrapText = true;
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 96;

for (const sheet of workbook.worksheets) {
  applyBasicStyle(sheet);
}

const inspect = await workbook.inspect({
  kind: "region",
  sheetId: "README",
  range: "A1:B8",
  maxChars: 3000,
});
console.log(inspect.ndjson);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

const preview = await workbook.render({
  sheetName: "README",
  range: "A1:B8",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(outputDir, "preview_issue_phrase_readme.png"), new Uint8Array(await preview.arrayBuffer()));

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`saved ${outputPath}`);
