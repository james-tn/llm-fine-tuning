import sqlite3
import json

def validate_sql_queries(input_file, db_path, output_file):
    valid_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         sqlite3.connect(db_path) as conn:

        conn.text_factory = str  # Handle TEXT/BLOB uniformly
        cursor = conn.cursor()

        for line in infile:
            total_count += 1
            data = json.loads(line)
            query = data["sql_result"].strip()

            try:
                cursor.execute(query)
                results = cursor.fetchall()

                # Check if exactly 1 record returned
                if len(results) == 1:
                    outfile.write(json.dumps(data) + '\n')
                    valid_count += 1
                else:
                    print(f"Invalid: Query returns {len(results)} rows")

            except sqlite3.Error as e:
                print(f"Invalid: SQL Error - {str(e)}")

    print(f"Valid queries: {valid_count}/{total_count}")

# Run validation
validate_sql_queries('sql_results.jsonl', 'northwind.db', 'sql_results_refined.jsonl')