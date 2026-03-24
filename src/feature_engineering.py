def add_features(df):
    """
    Adds engineered features to improve model performance.

    Parameters:
        df (pd.DataFrame): Cleaned dataframe

    Returns:
        pd.DataFrame: Dataframe with additional features
    """

    df = df.copy()

    # -------------------------
    # Total square footage
    # -------------------------
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # -------------------------
    # Age features
    # -------------------------
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # -------------------------
    # Total bathrooms
    # -------------------------
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # -------------------------
    # Interaction feature
    # -------------------------
    df["Qual_x_SF"] = df["OverallQual"] * df["GrLivArea"]

    return df