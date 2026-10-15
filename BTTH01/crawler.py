import requests
import pandas as pd
import json
import time

# API lay du lieu bat dong san Cho Tot (soi trong tab Network F12)
url = "https://gateway.chotot.com/v1/public/ad-listing"

data = []

# Lay 5 trang dữ liệu (100 tin)
for page in range(5):
    params = {
        'cg': 1000,          # Bat dong san
        'region_v2': 13000,   # TP.HCM
        'limit': 20,
        'o': page * 20
    }
    
    res = requests.get(url, params=params)
    res_json = res.json()
    ads = res_json['ads']
    
    for item in ads:
        title = item['subject'] if 'subject' in item else ''
        price = item['price'] if 'price' in item else 0
        area = item['size'] if 'size' in item else 0
        district = item['area_name'] if 'area_name' in item else ''
        region = item['region_name'] if 'region_name' in item else ''
        category = item['category_name'] if 'category_name' in item else ''
        body = item['body'] if 'body' in item else ''
        
        data.append({
            'title': title,
            'price': price,
            'area': area,
            'district': district,
            'region': region,
            'category': category,
            'body': body
        })
        
    print(f"Xong trang {page + 1}")
    time.sleep(1)

# Chuyen thanh DataFrame
df = pd.DataFrame(data)

# Luu file theo cac dinh dang de bai yeu cau
df.to_csv("data_bds.csv", index=False, encoding="utf-8")
df.to_csv("data_bds.tsv", sep="\t", index=False, encoding="utf-8")

with open("data_bds.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("Da luu xong cac file data_bds.csv, data_bds.tsv va data_bds.json!")