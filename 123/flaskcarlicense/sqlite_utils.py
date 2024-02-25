
import sqlite3
import json

DataBase='database/example.db'     
db = sqlite3.connect(DataBase,check_same_thread=False)
select_sql="SELECT * FROM PRODUCTS WHERE p_price >= 50"


def run_sql(db,sql):
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    return cursor
    
    
def query_db(db,sql):
    cursor=run_sql(db,sql)
    rows = cursor.fetchall()
    return rows

def query_db_json(db,sql):
    cursor=run_sql(db,sql)
    rows = cursor.fetchall()
    result=[]
    for row in rows:
        #print(row)
        item_template={} 
        item_template['pid']=row[0]
        item_template['p_category']=row[1]
        item_template['p_name']=row[2]
        item_template['p_price']=row[3]
        result.append(item_template)
    return json.dumps(result) 

if __name__ == '__main__':
    

    #ret=query_db(db,select_sql)
    ret=query_db_json(db,select_sql)
    print(ret)