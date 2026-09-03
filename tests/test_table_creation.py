from sqlalchemy import inspect
from src.setup_db import weatherDB, WeatherData


def test_table_creation(tmp_path):
    dburl = f"sqlite:///{tmp_path}/weather.db"
    db = weatherDB(dburl)

    inspector = inspect(db.engine)

    # table exists
    assert "weather_data" in inspector.get_table_names()

    # columns in the actual DB match the columns defined on the model
    actual_columns = {col["name"] for col in inspector.get_columns("weather_data")}
    expected_columns = {c.name for c in WeatherData.__table__.columns}
    assert actual_columns == expected_columns
