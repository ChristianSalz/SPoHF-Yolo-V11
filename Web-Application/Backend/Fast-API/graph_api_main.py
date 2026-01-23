from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later: restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

@app.get("/locations")
def get_locations():
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location)
            RETURN l.name AS name
            ORDER BY name
        """)
        return [r["name"] for r in res]

@app.get("/measurements")
def get_measurements(location: str, limit: int = 25):
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location {name:$loc})-[:HAS_MEASUREMENT]->(m:Measurement)
            RETURN
              toString(m.timestamp) AS timestamp,
              m.temperature          AS temperature,
              m.humidity             AS humidity,
              m.pressure             AS pressure,
              m.precipitation        AS precipitation,
              m.wind_speed           AS wind_speed
            ORDER BY m.timestamp DESC
            LIMIT $lim
        """, loc=location, lim=limit)

        out = []
        for r in res:
            out.append({
                "timestamp": r["timestamp"],
                "temperature": r["temperature"],
                "humidity": r["humidity"],
                "pressure": r["pressure"],
                "precipitation": r["precipitation"],
                "wind_speed": r["wind_speed"],
            })
        return list(reversed(out))

@app.get("/soil-measurements")
def get_soil_measurements(location: str, limit: int = 25):
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location {name:$loc})-[:HAS_MEASUREMENT]->(soil:SoilMeasurement)
            RETURN
              toString(soil.timestamp) AS timestamp,
              soil.soil_temp_0cm AS soil_temp_0cm,
              soil.soil_temp_6cm AS soil_temp_6cm,
              soil.soil_temp_18cm AS soil_temp_18cm,
              soil.soil_moisture_0_1cm AS soil_moisture_0_1cm,
              soil.soil_moisture_1_3cm AS soil_moisture_1_3cm,
              soil.soil_moisture_3_9cm AS soil_moisture_3_9cm
            ORDER BY soil.timestamp DESC
            LIMIT $lim
        """, loc=location, lim=limit)

        out = []
        for r in res:
            out.append({
                "timestamp": r["timestamp"],
                "soil_temp_0cm": r["soil_temp_0cm"],
                "soil_temp_6cm": r["soil_temp_6cm"],
                "soil_temp_18cm": r["soil_temp_18cm"],
                "soil_moisture_0_1cm": r["soil_moisture_0_1cm"],
                "soil_moisture_1_3cm": r["soil_moisture_1_3cm"],
                "soil_moisture_3_9cm": r["soil_moisture_3_9cm"],
            })
        return list(reversed(out))

@app.get("/solar-measurements")
def get_solar_measurements(location: str, limit: int = 25):
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location {name:$loc})-[:HAS_MEASUREMENT]->(sm:SunMeasurement)
            RETURN
              toString(sm.timestamp) AS timestamp,
              sm.uv_index            AS uv_index,
              sm.direct_radiation    AS direct_radiation
            ORDER BY sm.timestamp DESC
            LIMIT $lim
        """, loc=location, lim=limit)

        out = []
        for r in res:
            out.append({
                "timestamp": r["timestamp"],
                "uv_index": r["uv_index"],
                "direct_radiation": r["direct_radiation"],
            })
        return list(reversed(out))
    
    