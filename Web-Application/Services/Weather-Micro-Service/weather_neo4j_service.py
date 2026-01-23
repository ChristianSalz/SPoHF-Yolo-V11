import os
import time
import requests
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherNeo4jService:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_user = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')

        self.company_name = os.getenv('COMPANY_NAME', 'Vitarom Gbr')

        self.locations = [
            {"name": "Neurath",  "latitude": 51.0447, "longitude": 6.6847},
            {"name": "Straelen", "latitude": 51.4410, "longitude": 6.2620},
        ]

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )

        self.initialize_schema()

    def initialize_schema(self):
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                session.run("""
                    CREATE CONSTRAINT location_name IF NOT EXISTS
                    FOR (l:Location) REQUIRE l.name IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT company_name IF NOT EXISTS
                    FOR (c:Company) REQUIRE c.name IS UNIQUE
                """)
                logger.info("Schema initialized")
            except Exception as e:
                logger.warning(f"Schema creation error: {e}")

    # --------------------------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------------------------
    def fetch_weather_data(self, latitude: float, longitude: float, location_name: str):
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl",
                "timezone": "Europe/Berlin",
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            current = data.get('current', {})

            weather_data = {
                'timestamp': current.get('time'),
                'temperature': current.get('temperature_2m'),
                'humidity': current.get('relative_humidity_2m'),
                'precipitation': current.get('precipitation'),
                'wind_speed': current.get('wind_speed_10m'),
                'pressure': current.get('pressure_msl'),
                'location': location_name,
                'latitude': latitude,
                'longitude': longitude,
                'company': self.company_name,
            }

            logger.info(f"[{location_name}] Fetched weather: {weather_data}")
            return weather_data

        except Exception as e:
            logger.error(f"[{location_name}] Weather fetch error: {e}")
            return None

    def store_in_neo4j(self, weather_data: dict):
        if not weather_data:
            return
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                session.execute_write(self._create_weather_record, weather_data)
            logger.info(f"Stored weather data for {weather_data['location']}")
        except Exception as e:
            logger.error(f"Error storing weather: {e}")

    @staticmethod
    def _create_weather_record(tx, data: dict):
        query = """
        MERGE (c:Company {name: $company})
        MERGE (l:Location {name: $location})
          ON CREATE SET l.latitude = $latitude, l.longitude = $longitude
          ON MATCH  SET l.latitude = $latitude, l.longitude = $longitude
        MERGE (c)-[:HAS_LOCATION]->(l)

        CREATE (m:Measurement {
            timestamp: CASE WHEN $timestamp IS NULL THEN datetime() ELSE datetime($timestamp) END,
            temperature: $temperature,
            humidity: $humidity,
            precipitation: $precipitation,
            wind_speed: $wind_speed,
            pressure: $pressure
        })
        CREATE (l)-[:HAS_MEASUREMENT]->(m)
        RETURN m
        """
        return tx.run(query, **data).single()

    # --------------------------------------------------------------------------
    # SOLAR (UV & Direct Radiation Only)
    # --------------------------------------------------------------------------
    def fetch_sun_data(self, latitude: float, longitude: float, location_name: str):
        """Fetch UV + Direct Radiation only."""
        try:
            logger.info(f"[{location_name}] Starting sun data fetch...")
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "uv_index,direct_radiation",
                "timezone": "Europe/Berlin",
            }

            logger.info(f"[{location_name}] API Request params: {params}")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            logger.info(f"[{location_name}] API Response status: {response.status_code}")
            
            data = response.json()
            logger.info(f"[{location_name}] Raw API response: {data}")
            
            current = data.get("current", {})
            logger.info(f"[{location_name}] Current data extracted: {current}")

            sun_data = {
                "timestamp": current.get("time"),
                "uv_index": current.get("uv_index"),
                "direct_radiation": current.get("direct_radiation"),
                "location": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "company": self.company_name,
            }

            logger.info(f"[{location_name}] Formatted sun_data: {sun_data}")
            logger.info(f"[{location_name}] Timestamp value: {sun_data['timestamp']} (type: {type(sun_data['timestamp'])})")
            logger.info(f"[{location_name}] UV Index: {sun_data['uv_index']}")
            logger.info(f"[{location_name}] Direct Radiation: {sun_data['direct_radiation']}")

            return sun_data

        except Exception as e:
            logger.error(f"[{location_name}] Solar fetch error: {e}", exc_info=True)
            return None

    def store_sun_in_neo4j(self, sun_data: dict):
        if not sun_data:
            logger.warning("sun_data is None or empty, skipping storage")
            return
        
        logger.info(f"[{sun_data['location']}] Attempting to store solar data...")
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.execute_write(self._create_sun_record, sun_data)
                logger.info(f"[{sun_data['location']}] Stored solar data successfully: {result}")
        except Exception as e:
            logger.error(f"[{sun_data['location']}] Error storing solar data: {e}", exc_info=True)

    @staticmethod
    def _create_sun_record(tx, data: dict):
        query = """
        MERGE (c:Company {name: $company})
        MERGE (l:Location {name: $location})
          ON CREATE SET l.latitude = $latitude, l.longitude = $longitude
          ON MATCH  SET l.latitude = $latitude, l.longitude = $longitude
        MERGE (c)-[:HAS_LOCATION]->(l)

        CREATE (sm:SunMeasurement {
            timestamp: CASE WHEN $timestamp IS NULL THEN datetime() ELSE datetime($timestamp) END,
            uv_index: $uv_index,
            direct_radiation: $direct_radiation
        })
        CREATE (l)-[:HAS_MEASUREMENT]->(sm)
        RETURN sm
        """
        logger.info(f"Executing Neo4j query with data: {data}")
        result = tx.run(query, **data).single()
        logger.info(f"Neo4j query result: {result}")
        return result

    # --------------------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------------------
    def run(self, interval_minutes=15):
        logger.info(f"Starting for {self.company_name}")
        logger.info(f"Locations: {', '.join([l['name'] for l in self.locations])}")
        logger.info("VERSION: weather+solar enabled")

        while True:
            cycle_start = time.time()

            try:
                for loc in self.locations:

                    # WEATHER
                    logger.info(f"\n{'='*60}")
                    logger.info(f"FETCHING WEATHER FOR: {loc['name']}")
                    logger.info(f"{'='*60}")
                    
                    wd = self.fetch_weather_data(
                        latitude=loc["latitude"],
                        longitude=loc["longitude"],
                        location_name=loc["name"]
                    )
                    if wd:
                        self.store_in_neo4j(wd)
                    else:
                        logger.warning(f"Weather data fetch returned None for {loc['name']}")

                    # SOLAR
                    logger.info(f"\n{'='*60}")
                    logger.info(f"FETCHING SOLAR DATA FOR: {loc['name']}")
                    logger.info(f"{'='*60}")
                    
                    sd = self.fetch_sun_data(
                        latitude=loc["latitude"],
                        longitude=loc["longitude"],
                        location_name=loc["name"]
                    )
                    if sd:
                        logger.info(f"Solar data ready, storing for {loc['name']}...")
                        self.store_sun_in_neo4j(sd)
                    else:
                        logger.warning(f"Solar data fetch returned None for {loc['name']}")

                elapsed = time.time() - cycle_start
                wait = max(0, interval_minutes * 60 - elapsed)
                logger.info(f"\nWaiting {int(wait)} seconds...")
                time.sleep(wait)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(60)

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    service = WeatherNeo4jService()
    try:
        service.run(interval_minutes=15)
    finally:
        service.close()