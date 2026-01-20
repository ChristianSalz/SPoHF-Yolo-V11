
import os
import time
import requests
from datetime import datetime
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

        # Company to link locations to
        self.company_name = os.getenv('COMPANY_NAME', 'Vitarom Gbr')

        # --- NEW: support multiple locations ---
        # You can extend this list with more sites at any time.
        self.locations = [
            {"name": "Neurath",  "latitude": 51.0447, "longitude": 6.6847},
            {"name": "Straelen", "latitude": 51.4410, "longitude": 6.2620},
        ]

        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

        # Initialize schema
        self.initialize_schema()

    def initialize_schema(self):
        """Create constraints and indexes"""
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                # Unique Location names
                session.run("""
                    CREATE CONSTRAINT location_name IF NOT EXISTS
                    FOR (l:Location) REQUIRE l.name IS UNIQUE
                """)
                # Optional but nice to have: unique Company names
                session.run("""
                    CREATE CONSTRAINT company_name IF NOT EXISTS
                    FOR (c:Company) REQUIRE c.name IS UNIQUE
                """)
                logger.info("Schema initialized")
            except Exception as e:
                logger.warning(f"Schema already exists or error: {e}")

    def fetch_weather_data(self, latitude: float, longitude: float, location_name: str):
        """Fetch current weather from Open-Meteo API"""
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
                'timestamp': current.get('time'),                       # ISO string
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
            logger.error(f"[{location_name}] Error fetching weather data: {e}")
            return None

    def store_in_neo4j(self, weather_data: dict):
        """Store one weather snapshot into Neo4j."""
        if not weather_data:
            return
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                session.execute_write(self._create_weather_record, weather_data)
                logger.info(f"Stored weather data for {weather_data['location']}")
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")

    @staticmethod
    def _create_weather_record(tx, data: dict):
        """
        Create/merge company + location, then create measurement and relationships.
        Note: timestamp is stored as Neo4j datetime from ISO string.
        """
        query = """
        // Ensure company and location nodes exist and are linked
        MERGE (c:Company {name: $company})
        MERGE (l:Location {name: $location})
          ON CREATE SET l.latitude = $latitude, l.longitude = $longitude
          ON MATCH  SET l.latitude = $latitude, l.longitude = $longitude
        MERGE (c)-[:HAS_LOCATION]->(l)

        // Create a measurement and link it to the location
        CREATE (m:Measurement {
          timestamp: datetime($timestamp),
          temperature: $temperature,
          humidity: $humidity,
          precipitation: $precipitation,
          wind_speed: $wind_speed,
          pressure: $pressure
        })
        CREATE (l)-[:HAS_MEASUREMENT]->(m)
        RETURN m
        """
        result = tx.run(query, **data)
        return result.single()

    def run(self, interval_minutes=15):
        """Main loop — fetch and store for each configured location every N minutes"""
        logger.info(f"Starting weather service for company '{self.company_name}'")
        logger.info(f"Locations: {', '.join([l['name'] for l in self.locations])}")
        logger.info(f"Fetching data every {interval_minutes} minutes")

        while True:
            cycle_start = time.time()
            try:
                for loc in self.locations:
                    wd = self.fetch_weather_data(
                        latitude=loc["latitude"],
                        longitude=loc["longitude"],
                        location_name=loc["name"],
                    )
                    if wd:
                        self.store_in_neo4j(wd)

                # Sleep remaining time (so both locations are updated roughly every N minutes)
                elapsed = time.time() - cycle_start
                wait = max(0.0, interval_minutes * 60 - elapsed)
                logger.info(f"Waiting {int(wait)} seconds until next cycle...")
                time.sleep(wait)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    service = WeatherNeo4jService()
    try:
        service.run(interval_minutes=15)
    finally:
        service.close()
