
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse import csr_matrix

def missing_check(data):
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values)
    else:
        print("No missing values found.")

    # Check data types of 'order_id' and 'product_name'
    if data['order_id'].dtype == object and data['product_name'].dtype == object:
        print("Data types of 'order_id' and 'product_name' are correct.")
    else:
        print("Data types of 'order_id' and 'product_name' are incorrect.")
        # Convert 'order_id' to numeric type
        data['order_id'] = pd.to_numeric(data['order_id'], errors='coerce')

        # Convert 'product_name' to string type
        data['product_name'] = data['product_name'].astype(str)
        print("Data types of 'order_id' and 'product_name' is converted.")



    # Verify 'reordered' column contains numeric values
    reordered_values = data['reordered']
    if pd.to_numeric(reordered_values, errors='coerce').notnull().all():
        print("'reordered' column contains numeric values.")
    else:
        print("'reordered' column contains non-numeric values.")
        # Convert 'order_id' to numeric type
        data['reordered'] = pd.to_numeric(data['reordered'], errors='coerce')
        print("Data types of 'reordered' is converted.")


    return data

def matrix_conversion(basket):
    basket = basket.groupby(['order_id', 'product_name'])['reordered'].count().unstack().reset_index().fillna(0).set_index('order_id')
    basket = basket.apply(lambda x: x.map(encode_units))
    print(basket.head())
    return basket
   
def encode_units(x):
    return 1 if x > 0 else 0

def apriori_algorithm(basket):
    frequent_items = apriori(basket, min_support=0.005, use_colnames=True , verbose =1 , low_memory=True)

    # The length column has been added to increase ease of filtering.
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
    return frequent_items

def association(frequent_items):
    rules = association_rules(frequent_items, metric='lift', min_threshold=1.1)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
    final_pairs=rules[ (rules['antecedent_len'] >= 1) & (rules['confidence'] >= 0.15)]
    return final_pairs

def toSql(dataframe):
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password='root', database='capstone', charset='utf8')
        cur = conn.cursor()

        # Clear the existing data in the table
        cur.execute('DELETE FROM MB_pairs')

        # Iterate through the DataFrame rows and insert each row into the database
        for _, row in dataframe.iterrows():
            # Convert frozenset objects to normal strings
            antecedents = ', '.join(row[0])
            consequents = ', '.join(row[1])
            antecedent_support = float(row[2])
            consequent_support = float(row[3])
            support = float(row[4])
            confidence = float(row[5])
            lift = float(row[6])
            leverage = float(row[7])
            conviction = float(row[8])
            zhangs_metric = float(row[9])
            antecedent_len = int(row[10])
            consequents_len = int(row[11])

            # SQL query with parameterized values to avoid SQL injection
            query = "INSERT INTO mb_pairs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (antecedents, consequents, antecedent_support, consequent_support, support, confidence, lift, leverage, conviction, zhangs_metric, antecedent_len, consequents_len)
            cur.execute(query, values)

        print ("Working Good!!!")
        # Commit the changes to the database
        conn.commit()
        return 1
    finally:
        # Close cursor and connection
        if 'conn' in locals() and conn.is_connected():
            cur.close()
            conn.close()
        
def fromSql():
    try:
        # Establish a connection to the MySQL database
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='capstone', charset='utf8')

        # Define your SQL query
        query = "SELECT antecedents,consequents FROM MB_pairs"

        # Execute the query and fetch the results
        cursor = connection.cursor()
        cursor.execute(query)

        # Fetch all rows and store them in a list of tuples
        rows = cursor.fetchall()

        # Get the column names
        column_names = [i[0] for i in cursor.description]

        # Close cursor and connection
        cursor.close()
        connection.close()

        # Convert the data into a Pandas DataFrame
        data = pd.DataFrame(rows, columns=column_names)
        return data
    except mysql.connector.Error as error:
        print("An error occurred while connecting to the MySQL database:", error)
        # Handle the error or log it as needed
    except Exception as e:
        print("An unexpected error occurred:", e)
        # Handle the error or log it as needed
def final_products(result):
    List1=[]
    for _,row in result.iterrows():
        List1.append(row[0])
    List2=list(set(List1))
    return List2

def main():
   
    Data= pd.read_csv("D:\\Maaz's Document\\STUDY\\Capstone\\instacart\\Transection_Data.csv")

    Data=missing_check(Data)
    Matrix_Data = matrix_conversion(Data)
    """if Matrix_Data is not None:
        print(Matrix_Data.head())
    """
    Frequent_Products= apriori_algorithm(Matrix_Data)
    #print(Frequent_Products)

    Pairs=association(Frequent_Products)
    #print(Pairs)

        
    # Store the data to database
    store = toSql(Pairs)
    if store == 1:
        print ("Working Good!!!")
    else:
        print ("Error!!!")

    result = fromSql()
    #print (result)

    products= final_products(result)
    print(products)
    print(len(products))
    # Convert list to DataFrame
    d = pd.DataFrame(products, columns=['Column'])

    # File path to save CSV
    file_path = "D:\\Maaz's Document\\STUDY\\Capstone\\instacart\\Items.csv"

    # Write DataFrame to CSV
    d.to_csv(file_path, index=False)




if __name__=="__main__":
    main()
