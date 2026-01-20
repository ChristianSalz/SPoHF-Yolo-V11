
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
def get_measurements(location: str):
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location {name:$loc})-[:HAS_MEASUREMENT]->(m:Measurement)
            RETURN
              toString(m.timestamp) AS timestamp,   // <-- string now
              m.temperature          AS temperature,
              m.humidity             AS humidity,
              m.pressure             AS pressure,
              m.precipitation        AS precipitation,
              m.wind_speed           AS wind_speed
            ORDER BY m.timestamp
        """, loc=location)

        out = []
        for r in res:
            out.append({
                "timestamp": r["timestamp"],                 # already a string
                "temperature": r["temperature"],
                "humidity": r["humidity"],
                "pressure": r["pressure"],
                "precipitation": r["precipitation"],
                "wind_speed": r["wind_speed"],
            })
        return out
