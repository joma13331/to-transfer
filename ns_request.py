import requests
import pandas as pd

path_file = "D:\\Technical Education\\ineuron\\Internship\projects\\News Article Sorting\\NASCode\\NewsArticleSorting\\NAS_Artifact_Dir\\DataPreprocessed\\2022-09-12-10-07-48\\preprocessed_dataset\\preprocessde_train\\train_dataset.csv"

SERVICE_URL = "http://localhost:3000/predict"


def main():
    df = pd.read_csv(path_file)
    print(df.head())
    json_data = df.to_json()
    response = requests.post(
        SERVICE_URL,
        data=json_data,
        headers={"content-type": "application/json"}
    )
    print(response.text)


if __name__ == "__main__":
    main()
