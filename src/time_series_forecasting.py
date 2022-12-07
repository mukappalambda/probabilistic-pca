"""
Time Series Forecasting Using PPCA
"""
import pandas as pd

from ppca import PPCA


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)
    x = df[["Temp"]].to_numpy()
    x = x.reshape(-1, 365, order="C")
    dataset = x[:]
    latent_dim = 20
    ppca = PPCA(latent_dim)
    ppca.fit(dataset)
    print(ppca)
    size = 20
    samples = ppca.sample(size=size)
    print(samples)


if __name__ == "__main__":
    main()
