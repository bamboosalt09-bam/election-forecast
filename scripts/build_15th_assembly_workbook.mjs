import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "..");
const outputDir = path.join(repoRoot, "outputs", "15th_assembly_conversion");
const inputPath = path.join(outputDir, "15th_assembly_extracted.json");
const outputPath = path.join(outputDir, "제15대_국회_회의록_추출_데이터셋.xlsx");

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));

function cleanCell(value) {
  if (typeof value !== "string") {
    return value;
  }
  return value
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uD800-\uDFFF\uFFFE\uFFFF]/g, " ")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function cleanRows(rows) {
  return rows.map((row) => row.map((value) => cleanCell(value)));
}

function writeSheet(workbook, sheetName, columns, rows, options = {}) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const allRows = [columns, ...cleanRows(rows)];
  const chunkSize = options.chunkSize ?? 1000;
  for (let start = 0; start < allRows.length; start += chunkSize) {
    const chunk = allRows.slice(start, start + chunkSize);
    const range = sheet.getRangeByIndexes(start, 0, chunk.length, columns.length);
    range.values = chunk;
  }

  const used = sheet.getRangeByIndexes(0, 0, allRows.length, columns.length);
  used.format.font.name = "맑은 고딕";
  used.format.font.size = 10;
  used.format.wrapText = true;
  used.format.verticalAlignment = "top";

  const header = sheet.getRangeByIndexes(0, 0, 1, columns.length);
  header.format.fill.color = "#1F4E78";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  header.format.horizontalAlignment = "center";
  header.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sheet.freezePanes.freezeRows(1);

  const body = sheet.getRangeByIndexes(1, 0, Math.max(rows.length, 1), columns.length);
  body.format.borders = { preset: "insideHorizontal", style: "thin", color: "#E7E6E6" };

  for (let col = 0; col < columns.length; col += 1) {
    const columnName = String(columns[col]);
    const columnRange = sheet.getRangeByIndexes(0, col, allRows.length, 1);
    if (columnName.startsWith("발언내용")) {
      columnRange.format.columnWidth = 36;
    } else if (["source_file", "notes"].includes(columnName)) {
      columnRange.format.columnWidth = 42;
    } else if (["회의일자", "위원회", "안건", "발언자"].includes(columnName)) {
      columnRange.format.columnWidth = 18;
    } else {
      columnRange.format.columnWidth = 12;
    }
  }
  return sheet;
}

const workbook = Workbook.create();

writeSheet(workbook, "본회의_xlsx", payload.plenary_columns, payload.plenary_rows, { chunkSize: 800 });
writeSheet(workbook, "PDF_추출", payload.pdf_columns, payload.pdf_rows, { chunkSize: 500 });
writeSheet(workbook, "PDF_문서요약", payload.summary_columns, payload.summary_rows, { chunkSize: 200 });

const overview = workbook.worksheets.add("README");
overview.showGridLines = false;
const readmeRows = [
  ["항목", "값"],
  ["생성 목적", "제15대 국회 본회의 xlsx와 PDF zip 회의록을 통합 xlsx로 변환"],
  ["본회의_xlsx 행 수", payload.plenary_rows.length],
  ["PDF_추출 행 수", payload.pdf_rows.length],
  ["PDF 파일 수", payload.summary_rows.length],
  ["주의", "PDF_추출의 extraction_status가 garbled/empty인 행은 OCR 또는 수동 확인이 필요합니다."],
];
overview.getRangeByIndexes(0, 0, readmeRows.length, 2).values = readmeRows;
overview.getRange("A1:B1").format.fill.color = "#1F4E78";
overview.getRange("A1:B1").format.font.color = "#FFFFFF";
overview.getRange("A1:B1").format.font.bold = true;
overview.getRange("A:B").format.font.name = "맑은 고딕";
overview.getRange("A:B").format.wrapText = true;
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("B:B").format.columnWidth = 90;

await fs.mkdir(outputDir, { recursive: true });

const overviewInspect = await workbook.inspect({
  kind: "region",
  sheetId: "README",
  range: "A1:B6",
  maxChars: 3000,
});
console.log(overviewInspect.ndjson);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errorScan.ndjson);

const preview = await workbook.render({
  sheetName: "README",
  range: "A1:B6",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(outputDir, "preview_readme.png"), new Uint8Array(await preview.arrayBuffer()));

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`saved ${outputPath}`);
