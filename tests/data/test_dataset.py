import pytest
import tempfile
import os

from data.dataset import get_dataset


@pytest.fixture
def sample_data_files():
    # Create temporary files for testing
    r_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", encoding="utf-8")
    cs_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", encoding="utf-8")
    ds_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", encoding="utf-8")

    r_data = """Season,Round,Points,DriverID,ConstructorName
    2021,1,25,1,Alfa_Romeo F1 Team
    2021,2,18,2,Ferrari Team
    """
    cs_data = """season,round,points,constructorId
    2021,1,25,alfa_romeo
    2021,2,18,ferrari
    """
    ds_data = """season,round,points,driverId
    2021,1,25,1
    2021,2,18,2
    """

    r_file.write(r_data)
    cs_file.write(cs_data)
    ds_file.write(ds_data)

    r_file.close()
    cs_file.close()
    ds_file.close()

    yield r_file.name, cs_file.name, ds_file.name

    # Cleanup
    os.remove(r_file.name)
    os.remove(cs_file.name)
    os.remove(ds_file.name)


def test_get_dataset(sample_data_files):
    r_file, cs_file, ds_file = sample_data_files

    # Call the function
    result_df = get_dataset(r_file, cs_file, ds_file)

    # Expected columns in the resulting DataFrame
    expected_columns = [
        "season",
        "round",
        "points_x",
        "driver_id",
        "constructor_id",
        "season_to_date_constructor_points_per_round",
        "season_to_date_driver_points_per_round",
        "previous_constructor_points_per_round",
        "previous_driver_points_per_round",
    ]

    # Test the columns
    assert set(result_df.columns) == set(expected_columns)

    # Test if values are correct after processing
    assert result_df.loc[result_df["constructor_id"] == "alfa", "constructor_id"].values[0] == "alfa"
    assert result_df.loc[result_df["constructor_id"] == "ferrari", "constructor_id"].values[0] == "ferrari"
    assert result_df["season_to_date_constructor_points_per_round"].notna().all()
    assert result_df["season_to_date_driver_points_per_round"].notna().all()
    assert result_df["previous_constructor_points_per_round"].notna().all()
    assert result_df["previous_driver_points_per_round"].notna().all()

    # Test for any missing values in critical columns
    assert not result_df[["season", "round", "constructor_id", "driver_id"]].isnull().values.any()
