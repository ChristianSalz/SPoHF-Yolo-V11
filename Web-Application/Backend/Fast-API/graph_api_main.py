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