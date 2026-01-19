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
        
        # Neurath coordinates (Germany, near Cologne)
        self.latitude = 51.0447
        self.longitude = 6.6847
        self.location_name = "Neurath"
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize schema
        self.initialize_schema()
        
    def initialize_schema(self):
        """Create constraints and indexes"""
        with self.driver.session(database=self.neo4j_database) as session:
            # Create constraint for Location
            try:
                session.run("""
                    CREATE CONSTRAINT location_name IF NOT EXISTS
                    FOR (l:Location) REQUIRE l.name IS UNIQUE
                """)
                logger.info("Schema initialized")
            except Exception as e:
                logger.warning(f"Schema already exists or error: {e}")
    
    def fetch_weather_data(self):
        """Fetch current weather from Open-Meteo API"""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl",
                "timezone": "Europe/Berlin"
            }
            
            response = requests.get(url, params=params, timeout=10)
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
                'location': self.location_name,
                'latitude': self.latitude,
                'longitude': self.longitude
            }
            
            logger.info(f"Fetched weather data: {weather_data}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def store_in_neo4j(self, weather_data):
        """Store weather data in Neo4j graph database"""
        if not weather_data:
            return
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                session.execute_write(self._create_weather_record, weather_data)
                logger.info(f"Stored weather data in Neo4j for {weather_data['location']}")
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
    
    @staticmethod
    def _create_weather_record(tx, data):
        """Create weather measurement node and relationships"""
        query = """
        MERGE (l:Location {name: $location})
        SET l.latitude = $latitude,
            l.longitude = $longitude
        
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
    
    def run(self, interval_minutes=10):
        """Main loop - fetch and store data every N minutes"""
        logger.info(f"Starting weather service for {self.location_name}")
        logger.info(f"Fetching data every {interval_minutes} minutes")
        
        while True:
            try:
                weather_data = self.fetch_weather_data()
                if weather_data:
                    self.store_in_neo4j(weather_data)
                
                logger.info(f"Waiting {interval_minutes} minutes until next fetch...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("Connection closed")

if __name__ == "__main__":
    service = WeatherNeo4jService()
    try:
        service.run(interval_minutes=10)
    finally:
        service.close()