from flask import Flask, request, render_template, session, redirect, url_for
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

app = Flask(__name__ , static_url_path='/static')
app.secret_key = 'your_secret_key_here'

# Define credentials
USERNAME = 'root'
PASSWORD = 'root'


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
    print(frequent_items)
    return frequent_items

def association(frequent_items):
    rules = association_rules(frequent_items, metric='lift', min_threshold=1.1)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
    final_pairs=rules[ (rules['antecedent_len'] >= 1) & (rules['confidence'] >= 0.15)]
    return final_pairs,rules

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
 
def toSql_allpairs(dataframe):
    try:
        conn = mysql.connector.connect(host='localhost', user='root', password='root', database='capstone', charset='utf8')
        cur = conn.cursor()

        # Clear the existing data in the table
        cur.execute('DELETE FROM mb_allpairs')

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
            query = "INSERT INTO mb_allpairs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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



def fromSql_allpairs():
    try:
        # Establish a connection to the MySQL database
        connection = mysql.connector.connect(host='localhost', user='root', password='root', database='capstone', charset='utf8')

        # Define your SQL query
        query = "SELECT consequents, antecedents, COUNT(*) AS pair_count FROM ( SELECT LEAST(consequents, antecedents) AS consequents, GREATEST(consequents, antecedents) AS antecedents FROM MB_allpairs) AS paired_items GROUP BY consequents, antecedents ORDER BY pair_count DESC;"

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
    for i in range(0,len(List2)):
        List2[i]=List2[i].replace(' ', '_')
      
    # Create an empty list to store the split values
    split_values = []

    # Iterate through each item in the original list
    for item in List2:
        # Split the item by ','
        split_values.extend(item.split(','))
    unique_values_list1 = [value.lstrip('Organic_') for value in split_values]
    unique_values_list = [value.lstrip('_') for value in unique_values_list1]
    # Convert the list to a set to remove duplicates
    unique_values_set = set(unique_values_list)
    values_list = list(unique_values_set)
    Links=[]
    for item in values_list:
        Links.append("{}.jpg".format(item))
    print(Links)
    return 1




@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', message=session.pop('message', None))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/retrieve_pairs')
def retrieve_pairs():
    result = fromSql_allpairs()  # Assuming fromSql_allpairs() returns a DataFrame
    
    print('---------')
    print(result)
    print('---------')
    return render_template('pairs.html', result=result)



@app.route('/Frequency')
def Frequency():

    df = pd.read_csv("D:\Maaz's_Document\STUDY\Capstone\instacart\Capstone\Transection_Data.csv")
    # Clustering (K-means)
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df.drop(['order_id', 'product_name'], axis=1))

    # Frequency analysis
    product_freq = df.groupby('cluster').sum().drop(['order_id', 'add_to_cart_order', 'reordered', 'aisle_id', 'department_id'], axis=1)
    
    # Frequency analysis
    product_freq = df['product_name'].value_counts()

    return render_template('Frequency.html', result=product_freq)




@app.route('/upload', methods=['POST'])

def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        session['message'] = 'No selected file'
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    if file.filename == '':
        session['message'] = 'No selected file'
        return redirect(url_for('index'))
    
    if file:
        # Process the DataFrame here
        Data= pd.read_csv(file)

        Data=missing_check(Data)
        Matrix_Data = matrix_conversion(Data)
        """if Matrix_Data is not None:
            print(Matrix_Data.head())
        """
        Frequent_Products= apriori_algorithm(Matrix_Data)
        #print(Frequent_Products)

        Pairs,all_pairs=association(Frequent_Products)
        #print(Pairs)
        print("All Pairs")
        print(all_pairs)
        print("Pairs")
        print(Pairs)
        store = toSql_allpairs(all_pairs)
        
        #Store the data to database

        store = toSql(Pairs)
        if store == 1:
            print ("Working Good!!!")
        else:
            print ("Error!!!")

        result = fromSql()
        #print (result)

        products= final_products(result)
        
        return render_template('index.html', image_value=products)

if __name__ == '__main__':
    app.run(debug=True)



