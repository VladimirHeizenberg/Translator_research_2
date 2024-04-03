import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

language = "en-ru"
url = f"https://data.statmt.org/opus-100-corpus/v1.0/supervised/{language}/"
save_directory = f"./Datasets/{language}"

os.makedirs(save_directory, exist_ok=True)

response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

links = soup.find_all('a')

file_links = [link['href'] for link in links if '.' in link['href']]

for file_link in tqdm(file_links):
    file_url = url + file_link
    save_path = os.path.join(save_directory, file_link)
    
    print(f"Downloading {file_url}")
    
    file_response = requests.get(file_url)
    if file_response.status_code == 404:
        print(f"Could not download {file_url}")
        continue
    
    with open(save_path, 'wb') as file:
        file.write(file_response.content)
    
    print(f"Saved {file_link}")

print("All files have been downloaded.")