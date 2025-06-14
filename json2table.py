import json
import os
import csv
import re
from pathlib import Path

def extract_tables_with_smart_titles(blocks, out_dir="output_tables"):
    os.makedirs(out_dir, exist_ok=True)
    tables = []
    previous_titles = []
    table_counter = 1

    def clean_title(title, max_length=60):
        if not isinstance(title, str):
            title = str(title or "Untitled")

        # Allow Arabic, Latin, numbers, spaces, and dash
        title = re.sub(r"[^\w\u0600-\u06FF\s\-]", "", title)
        title = re.sub(r"[\s\-]+", " ", title).strip()

        # Trim title
        if len(title) > max_length:
            title = title[:max_length].rstrip()

        return title or "Untitled"

    def find_best_title(current_index):
        # Look backwards for a meaningful block (not table)
        for i in range(current_index - 1, -1, -1):
            block = blocks[i]
            if block["type"] in ["Caption", "Title", "Section-header", "Section-subheader", "Page-header"]:
                text = block.get("text_representation")
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped and not stripped.isdigit():
                        return stripped
        return "Untitled"

    for i, block in enumerate(blocks):
        if block.get("type") == "table":
            table_data = block.get("table", {}).get("cells", [])
            if not table_data:
                continue

            # Extract content from table into a grid
            rows_dict = {}
            for cell in table_data:
                row_idx = cell["rows"][0]
                col_idx = cell["cols"][0]
                content = cell.get("content", "").strip()
                rows_dict.setdefault(row_idx, {})[col_idx] = content

            # Convert to list of lists, fill missing cells with ""
            max_row = max(rows_dict)
            max_col = max([max(row.keys()) for row in rows_dict.values()])
            table_rows = []
            for r in range(max_row + 1):
                row = []
                for c in range(max_col + 1):
                    row.append(rows_dict.get(r, {}).get(c, ""))
                table_rows.append(row)

            # Find title
            raw_title = find_best_title(i)
            cleaned_title = clean_title(raw_title)
            numbered_title = f"{table_counter:02d}_{cleaned_title}"
            filename = f"{numbered_title}.csv"

            # Avoid duplicate names
            while filename in previous_titles:
                table_counter += 1
                numbered_title = f"{table_counter:02d}_{cleaned_title}"
                filename = f"{numbered_title}.csv"

            previous_titles.append(filename)
            tables.append((filename, table_rows))
            table_counter += 1

    # Save all tables
    for filename, rows in tables:
        with open(os.path.join(out_dir, filename), mode="w", newline='', encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"✅ Extracted and saved {len(tables)} tables to: {out_dir}")



if __name__ == "__main__":
    base_json_dir = "Statements"
    base_output_dir = "parsed_tables"

    # Process each country directory
    for country_dir in os.listdir(base_json_dir):
        country_path = os.path.join(base_json_dir, country_dir)
        if not os.path.isdir(country_path) or country_dir.startswith('.'):
            continue

        # Create country-specific output directory
        country_output_dir = os.path.join(base_output_dir, country_dir)
        os.makedirs(country_output_dir, exist_ok=True)

        # Process each company directory in the country directory
        for company_dir in os.listdir(country_path):
            company_path = os.path.join(country_path, company_dir)
            if not os.path.isdir(company_path) or company_dir.startswith('.'):
                continue

            # Create company-specific output directory
            company_output_dir = os.path.join(country_output_dir, company_dir)
            os.makedirs(company_output_dir, exist_ok=True)

            # Look for JSON files in the json subdirectory
            json_dir = os.path.join(company_path, "json")
            if not os.path.exists(json_dir):
                print(f"⚠️ No json directory found in {company_path}")
                continue

            # Process each JSON file in the json directory
            for filename in os.listdir(json_dir):
                if filename.endswith(".json"):
                    json_path = os.path.join(json_dir, filename)
                    name_without_ext = os.path.splitext(filename)[0]
                    output_dir = os.path.join(company_output_dir, name_without_ext)

                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Processing: {json_path} → {output_dir}")

                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            blocks = json.load(f)
                        extract_tables_with_smart_titles(blocks, out_dir=output_dir)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to load {filename}: {e}")
                    except Exception as e:
                        print(f"⚠️ Error processing {filename}: {e}")