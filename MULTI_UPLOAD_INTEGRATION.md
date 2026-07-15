# Multi-File Upload — Integration Notes

This update adds glob-based ingestion to the FastAPI webapp. Multiple CSV files
can now be uploaded at once and the dashboard automatically picks up whatever
sits in `data/uploads/` and `data/raw/`.

## What changed

| File | Status | Purpose |
|---|---|---|
| `webapp/ingestion.py` | **NEW** | `ingest_multi_glob()` + canonical patterns/schema |
| `webapp/main.py` | Patched | Upload route accepts multiple files; `load_raw_data()` rewired |
| `webapp/templates/upload.html` | Patched | `multiple` attribute on file input, copy updated |
| `webapp/static/css/style.css` | (unchanged) | Already from previous update |

## How to install

Drop these three files in over the existing ones — no other changes required.

```
zimsec_project/
├── webapp/
│   ├── ingestion.py                ← NEW
│   ├── main.py                     ← REPLACE
│   ├── templates/
│   │   └── upload.html             ← REPLACE
│   └── ...
```

## What happens now

### When the user uploads files via the dashboard

1. They select one **or many** CSV files in the upload widget.
2. Each file is saved to `data/uploads/` with a timestamp prefix (no collisions).
3. The route calls `ingest_multi_glob()` across:
   - `data/zimsec_olevel_district_data.csv` (master)
   - `data/raw/*.csv`
   - `data/uploads/*.csv`
   - `data/**/zimsec_*.csv`
4. Files missing required columns (`District`, `Province`, `Year`, `Setting`, `Pass_Rate_Pct`) are **silently skipped** — no crash.
5. Rows are deduplicated on `(District, Year, Setting)` keeping the last (newest) version when "Remove duplicate rows automatically" is ticked.
6. The upload page shows the combined preview, and a message like:
   > *Uploaded 2 file(s) — 20 new row(s) added. Combined dataset now has 1,440 records.*

### When the dashboard, analysis, or reports pages load

`load_raw_data()` now uses the same multi-glob ingestion. **The dashboard immediately reflects whatever is in the uploads folder** — no need to re-trigger anything.

### When a file fails validation

Two failure modes:

- **Wrong extension** (e.g., `.xlsx` selected when "CSV" is the file type) → file is rejected with a per-file reason shown in the message.
- **Missing required columns** → file is saved to `data/uploads/` but skipped at ingestion. The combined preview still works using the valid files.

## Schema requirements

Any CSV the system ingests must have these columns:

```
District, Province, Year, Setting, Pass_Rate_Pct
```

Extra columns (e.g., `Pupil_Teacher_Ratio`, `ICT_Resource_Index`) are kept if present and used by the model. Files with only some of the extra columns will work for ingestion but may produce `NaN` values in modeling — fix this with imputation if needed.

## Edge cases handled

- **Same file matched by multiple patterns** (e.g., master file matches both pattern 1 and pattern 4) — deduplicated at the path level before reading.
- **Empty upload directory** — falls back gracefully to just the master file.
- **No files anywhere** — `load_raw_data()` returns `None`, dashboard shows "no data" placeholder.
- **Mixed CSV + JSON in one upload** — JSON path is single-file only; CSVs go through the multi-glob.
- **Filename collisions in upload dir** — timestamp prefix includes microseconds (`%Y%m%d_%H%M%S_%f`), so even rapid simultaneous uploads don't collide.

## Rolling back

If you ever want to revert to single-file uploads, the only essential rollback step is to revert the `<input>` tag in `upload.html`:

```html
<!-- before rollback -->
<input ... name="files" ... multiple required>
<!-- after rollback -->
<input ... name="file" ... required>
```

…and rename the route parameter from `files: list[UploadFile]` back to `file: UploadFile`. The ingestion module can stay — it's only used by `load_raw_data()` and only requires the master file to function.

## Quick smoke test

Drop a 1-row test file in `data/uploads/`:

```csv
District,Province,Year,Setting,Pass_Rate_Pct,Number_of_Students,Number_of_Schools,Pupil_Teacher_Ratio,Textbook_Availability_Index,ICT_Resource_Index,School_Infrastructure_Index,Proportion_Schools_With_Science_Labs,Poverty_Incidence_Pct,Adult_Literacy_Rate_Pct,Unemployment_Rate_Pct,Avg_Household_Income_USD,Youth_Population_Density,Rural_Urban,Distance_to_Exam_Center_km,Previous_Year_Pass_Rate_Pct,Grade7_Pass_Rate_Pct
TestDistrict,TestProvince,2025,November,80.5,5000,50,30,0.7,0.7,0.8,0.6,40,90,20,500,400,Urban,5,79.0,75.0
```

Reload the dashboard — the "Total Records" stat should jump by 1, and "Year Range" should now show 2015–2025.
