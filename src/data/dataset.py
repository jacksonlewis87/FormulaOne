import pandas as pd


def get_dataset(r_file: str, cs_file: str, ds_file: str) -> pd.DataFrame:
    # results
    df_r = pd.read_csv(r_file)[["Season", "Round", "Points", "DriverID", "ConstructorName", "Grid"]]
    df_r.rename(
        columns={
            "Season": "season",
            "Round": "round",
            "Points": "points",
            "DriverID": "driver_id",
            "ConstructorName": "constructor_id",
            "Grid": "grid",
        },
        inplace=True,
    )
    df_r["constructor_id"] = df_r["constructor_id"].str.lower()
    df_r["constructor_id"] = df_r["constructor_id"].str.replace(" f1", "", regex=False)
    df_r["constructor_id"] = df_r["constructor_id"].str.replace(" team", "", regex=False)
    df_r["constructor_id"] = df_r["constructor_id"].str.replace(" ", "_", regex=False)
    df_r["constructor_id"] = df_r["constructor_id"].str.replace(r"^alfa_romeo$", "alfa", regex=True)

    # constructor standings
    df_cs = pd.read_csv(cs_file)[["season", "round", "points", "constructorId"]]
    df_cs.rename(columns={"constructorId": "constructor_id"}, inplace=True)

    # driver standings
    df_ds = pd.read_csv(ds_file)[["season", "round", "points", "driverId"]]
    df_ds.rename(columns={"driverId": "driver_id"}, inplace=True)

    # final constructor standings for each season
    max_round = df_cs.groupby("season")["round"].max()
    max_round.reset_index()
    max_round.columns = ["season", "round"]
    season_final_constructor_standings = pd.merge(df_cs, max_round, on=["season", "round"], how="inner")

    if len(season_final_constructor_standings) != len(df_cs.groupby(["season", "constructor_id"]).size()):
        raise ValueError("Missing constructor in final standings")

    # final driver standings for each season
    max_round = df_ds.groupby("season")["round"].max()
    max_round.reset_index()
    max_round.columns = ["season", "round"]
    season_final_driver_standings = pd.merge(df_ds, max_round, on=["season", "round"], how="inner")

    if len(season_final_driver_standings) != len(df_ds.groupby(["season", "driver_id"]).size()):
        raise ValueError("Missing driver in final standings")

    # create merged dataset
    df_cs["points"] = df_cs["points"] / df_cs["round"]
    df_cs["points"] = df_cs["points"].round(3)
    df_cs["round"] = df_cs["round"] + 1  # offset round
    df_cs.rename(columns={"points": "season_to_date_constructor_points_per_round"}, inplace=True)
    df_r = pd.merge(df_r, df_cs, on=["season", "round", "constructor_id"], how="left")

    df_ds["points"] = df_ds["points"] / df_ds["round"]
    df_ds["points"] = df_ds["points"].round(3)
    df_ds["round"] = df_ds["round"] + 1  # offset round
    df_ds.rename(columns={"points": "season_to_date_driver_points_per_round"}, inplace=True)
    df_r = pd.merge(df_r, df_ds, on=["season", "round", "driver_id"], how="left")

    season_final_constructor_standings["points"] = (
        season_final_constructor_standings["points"] / season_final_constructor_standings["round"]
    )
    season_final_constructor_standings["points"] = season_final_constructor_standings["points"].round(3)
    season_final_constructor_standings.drop(columns=["round"], inplace=True)
    season_final_constructor_standings["season"] = season_final_constructor_standings["season"] + 1  # offset season
    season_final_constructor_standings.rename(columns={"points": "previous_constructor_points_per_round"}, inplace=True)
    df_r = pd.merge(df_r, season_final_constructor_standings, on=["season", "constructor_id"], how="left")

    season_final_driver_standings["points"] = (
        season_final_driver_standings["points"] / season_final_driver_standings["round"]
    )
    season_final_driver_standings["points"] = season_final_driver_standings["points"].round(3)
    season_final_driver_standings.drop(columns=["round"], inplace=True)
    season_final_driver_standings["season"] = season_final_driver_standings["season"] + 1  # offset season
    season_final_driver_standings.rename(columns={"points": "previous_driver_points_per_round"}, inplace=True)
    df_r = pd.merge(df_r, season_final_driver_standings, on=["season", "driver_id"], how="left")

    return df_r
