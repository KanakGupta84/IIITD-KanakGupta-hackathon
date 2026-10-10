import requests

url = "https://api.gdeltproject.org/api/v2/doc/doc"

params = {
    "query": 'sourcecountry:india sourcelang:english (RBI OR "interest rates" OR inflation OR banking OR markets OR stocks OR "credit rating")',
    "mode": "artlist",
    "maxrecords": 5,
    "timespan": "1day",
    "format": "json",
    "sort": "datedesc"
}

response = requests.get(url, params=params, timeout=30)

print("Status code:", response.status_code)

data = response.json()

print("Number of articles:", len(data.get("articles", [])))

for article in data.get("articles", []):
    print("\nTitle:", article.get("title"))
    print("URL:", article.get("url"))
    print("Date:", article.get("seendate"))