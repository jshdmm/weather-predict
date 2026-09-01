from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine, orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
import pandas as pd

# Define the base class for declarative class definitions
Base = orm.declarative_base()


# Define the WeatherData table structure
class WeatherData(Base):
    __tablename__ = 'weather_data'
    id = Column(Integer, primary_key=True)
    time = Column(DateTime, index=True)
    temperature_2m = Column(Float)
    relativehumidity_2m = Column(Float)
    rain = Column(Float)
    snowfall = Column(Float)
    windspeed_10m = Column(Float)
    winddirection_10m = Column(Float)
    soil_temperature_0_to_7cm = Column(Float)

# Define the weather database class for database operations
class weatherDB:
    def __init__(self, db_url):
        # Initialize the database engine and create tables if they don't exist
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


    # Method to add weather data to the database
    def add_weather_data(self, data):
        with orm.Session(self.engine) as session:
            try:
                for record in data:
                    record = dict(record)
                    # The Open-Meteo API returns 'time' as unix seconds (UTC);
                    # convert it to a naive UTC datetime for the DateTime column.
                    if isinstance(record['time'], (int, float)):
                        record['time'] = datetime.fromtimestamp(
                            record['time'], tz=timezone.utc).replace(tzinfo=None)
                    session.add(WeatherData(**record))
                session.commit()
            except Exception as e:
                session.rollback()
                raise Exception(f"Error inserting weather data: {e}")


    # Method to retrieve weather data from the database
    def get_weather_data(self):
        with orm.Session(self.engine) as session:
            query = session.query(WeatherData).all()
        data = [(row.id, row.time, row.temperature_2m, row.relativehumidity_2m, row.rain, row.snowfall, row.windspeed_10m, row.winddirection_10m, row.soil_temperature_0_to_7cm) for row in query]
        df = pd.DataFrame(data, columns=['id', 'time', 'temperature_2m', 'relativehumidity_2m', 'rain', 'snowfall', 'windspeed_10m', 'winddirection_10m', 'soil_temperature_0_to_7cm'])
        return df
