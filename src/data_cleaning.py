def clean_data(df):
    """
    Handles missing values and prepares dataset for modeling

    Parameters:
        df (pd.DataFrame): Raw input dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    df = df.copy()

    # ------------------------------
    # Categorical: missing = absence
    # ------------------------------
    none_cols = [
        "PoolQC", "Alley", "Fence", "FireplaceQu", 
        "GarageCond", "GarageType", "GarageFinish", "GarageQual",
        "BsmtFinType1", "BsmtFinType2", "BsmtExposure", "BsmtCond", "BsmtQual",
        "MasVnrType"
        ]
    # prevents crashing in case of schema change
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # ------------------------------
    # Numerical: missing = 0 (no feature)
    # ------------------------------
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    # ------------------------------
    # Numerical: grouped imputation
    # ------------------------------
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    ) 

    # ------------------------------
    # Categorical: small missing -> mode
    # ------------------------------
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # ------------------------------
    # Drop low-value feature
    # ------------------------------
    df = df.drop(columns=["MiscFeature"])   

    return df