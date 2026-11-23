#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path(
    "/Users/louis/Desktop/Tronc commun 2A/Projet S7/AI_and_Biodiversity/raw_data/"
    "STOC_pressions_bio_assol_meteo_pesti_routes_lum_120625.csv"
)
# Alternative :
# "STOC_pressions_bio_assol_meteo_pesti_140525.csv"


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of these columns found: {candidates}\nAvailable columns:\n{list(df.columns)}"
    )


def main():
    # IMPORTANT: separator is ';'
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")

    year_col = pick_col(df, ["Year", "year", "ANNEE", "annee"])
    abundance_col = pick_col(df, ["abondance_capped", "abondance", "Abondance"])

    # Clean
    df = df[[year_col, abundance_col]].copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[abundance_col] = pd.to_numeric(df[abundance_col], errors="coerce")
    df = df.dropna(subset=[year_col, abundance_col])

    # Aggregate
    total_by_year = (
        df.groupby(year_col, as_index=False)[abundance_col]
          .sum()
          .sort_values(year_col)
          .reset_index(drop=True)
    )

    # Plot total
    plt.figure(figsize=(10, 6))
    plt.plot(total_by_year[year_col], total_by_year[abundance_col], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Total abundance (all species, all zones)")
    plt.title("Evolution of total bird abundance (STOC data)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot index base 100
    base = total_by_year[abundance_col].iloc[0]
    if base > 0:
        total_by_year["index_100"] = total_by_year[abundance_col] / base * 100

        plt.figure(figsize=(10, 6))
        plt.plot(total_by_year[year_col], total_by_year["index_100"], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Index (base 100)")
        plt.title("Evolution of total bird abundance (index base 100)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(total_by_year)


if __name__ == "__main__":
    main()
