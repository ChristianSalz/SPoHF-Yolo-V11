
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # optionally restrict to your frontend domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

@app.get("/locations")
def get_locations():
    with driver.session() as session:
        res = session.run("MATCH (l:Location) RETURN l.name AS name ORDER BY name")
        return [r["name"] for r in res]

@app.get("/measurements")
def get_measurements(location: str):
    with driver.session() as session:
        res = session.run("""
            MATCH (l:Location {name:$loc})-[:HAS_MEASUREMENT]->(m:Measurement)
            RETURN m ORDER BY m.timestamp
        """, loc=location)
        out = []
        for r in res:
            m = r["m"]
            out.append({
                "timestamp": m["timestamp"],
                "temperature": m.get("temperature"),
                "humidity": m.get("humidity"),
                "pressure": m.get("pressure"),
                "precipitation": m.get("precipitation"),
                "wind_speed": m.get("wind_speed"),
            })
        return out
