from dengue_prediction.dataset.dengue_preprocess import dengue_preprocess

if __name__ == "__main__":
    csv_file = 'Dengue_Daily.csv'
    dengue_preprocess(csv_file, period='monthly')
