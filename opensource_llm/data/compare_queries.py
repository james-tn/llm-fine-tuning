import sqlite3

def compare_sql_results(predicted_sql_query, ground_truth_query, db_path='northwind.db'):
    def _execute_and_format(sql_query):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                if len(result) != 1:
                    return None
                return sorted(map(str, result[0]))  # Sort values as strings
            except sqlite3.Error:
                return None

    # Get normalized results
    pred_result = _execute_and_format(predicted_sql_query)
    truth_result = _execute_and_format(ground_truth_query)

    # Compare sorted value lists
    return (pred_result is not None) and \
           (truth_result is not None) and \
           (pred_result == truth_result)

# Example usage
predicted_sql_query = "SELECT \n    p.ProductName,\n    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / SUM(od.Quantity) AS avg_revenue_per_unit\nFROM \n    order_details od\nJOIN \n    Products p ON od.ProductID = p.ProductID\nGROUP BY \n    p.ProductID, p.ProductName\nORDER BY \n    avg_revenue_per_unit DESC\nLIMIT 1;"
ground_truth_query = "SELECT \n SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / SUM(od.Quantity) AS avg_revenue_per_unit_x,   p.ProductName as Name\nFROM \n    order_details od\nJOIN \n    Products p ON od.ProductID = p.ProductID\nGROUP BY \n    p.ProductID, p.ProductName\nORDER BY \n    avg_revenue_per_unit_x DESC\nLIMIT 1;"
result = compare_sql_results(predicted_sql_query, ground_truth_query)
print(result)