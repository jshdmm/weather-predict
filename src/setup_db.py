from sqlalchemy import Column, Integer, Float, DateTime, create_engine, orm
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
import pandas as pd

# Define the base class for declarative class definitions
Base = orm.declarative_base()


# Define the WeatherData table structure
class WeatherData(Base):
    __tablename__ = 'weather_data'
    id = Column(Integer, primary_key=True)
    time = Column(DateTime, unique=True, index=True)
    temperature_2m = Column(Float)
    relativehumidity_2m = Column(Float)
    rain = Column(Float)
    snowfall = Column(Float)
    windspeed_10m = Column(Float)
    winddirection_10m = Column(Float)
    soil_temperature_0_to_7cm = Column(Float)

# Define the weather database class for database operations
class weatherDB:
    def __init__(self, db_url: str) -> None:
        # Initialize the database engine and create tables if they don't exist
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


    # Method to add weather data to the database
    def add_weather_data(self, data: list) -> None:
        with orm.Session(self.engine) as session:
            try:
                for record in data:
                    record = dict(record)
                    # The Open-Meteo API returns 'time' as unix seconds (UTC);
                    # convert it to a naive UTC datetime for the DateTime column.
                    if isinstance(record['time'], (int, float)):
                        record['time'] = datetime.fromtimestamp(
                            record['time'], tz=timezone.utc).replace(tzinfo=None)
                    # Upsert by time: one row per timestamp. Re-running a
                    # fetch over an overlapping date range (e.g. the rolling
                    # 2-year backfill) should refresh existing rows in place
                    # rather than duplicate them.
                    existing = session.query(WeatherData).filter(
                        WeatherData.time == record['time']).first()
                    if existing is None:
                        session.add(WeatherData(**record))
                    else:
                        for key, value in record.items():
                            if key == 'time':
                                continue
                            setattr(existing, key, value)
                session.commit()
            except Exception as e:
                session.rollback()
                raise Exception(f"Error inserting weather data: {e}")


    # Method to retrieve weather data from the database
    def get_weather_data(self) -> pd.DataFrame:
        with orm.Session(self.engine) as session:
            query = session.query(WeatherData).all()
        data = [(row.id, row.time, row.temperature_2m, row.relativehumidity_2m, row.rain, row.snowfall, row.windspeed_10m, row.winddirection_10m, row.soil_temperature_0_to_7cm) for row in query]
        df = pd.DataFrame(data, columns=['id', 'time', 'temperature_2m', 'relativehumidity_2m', 'rain', 'snowfall', 'windspeed_10m', 'winddirection_10m', 'soil_temperature_0_to_7cm'])
        return df
