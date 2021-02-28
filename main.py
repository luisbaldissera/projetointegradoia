import json
import pandas as pd

import start

def main():
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    df = pd.read_csv(settings.get('file.csv'))
    start.start(df, settings)

if __name__ == "__main__":
    main()
