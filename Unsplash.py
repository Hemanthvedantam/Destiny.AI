import requests
import json
import time

UNSPLASH_ACCESS_KEY = "Cc1ctqZjt2-ApMzFiBhrudnBT097az20ndWcfD01UaE"

def get_image_url(query):
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page=1&client_id={UNSPLASH_ACCESS_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['urls']['regular']  # You can also use 'small' or 'thumb'
    return ""

def update_json_with_images(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        destinations = json.load(f)

    for dest in destinations:
        dest_name = dest['name']
        print(f"🔍 Fetching image for: {dest_name}")
        image_url = get_image_url(dest_name)
        dest['image'] = image_url
        time.sleep(1)  # Respect Unsplash API rate limits

    with open('data/destinations_with_images.json', 'w', encoding='utf-8') as f:
        json.dump(destinations, f, indent=2)

    print("✅ All images added using Unsplash!")

# Run the update
update_json_with_images('data/destinations.json')
