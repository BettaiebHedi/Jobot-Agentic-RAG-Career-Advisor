from pymilvus import connections, Collection

connections.connect(uri="tcp://localhost:19530")

collections = [
    "freelance_hybrid",
    "freelance_tunisia_hybrid",
    "startup_tunisia_hybrid",
    "startup_hybrid",
    "certifications_hybrid",
    "Code_de_travail_hybrid",
    "cv_enhancement_hybrid",
    "international_labor_market_hybrid",
    "tunisian_labor_market_hybrid"
]

for name in collections:
    try:
        c = Collection(name)
        c.load()
        print(f"{name}: {c.num_entities} entities")
    except Exception as e:
        print(f"{name}: not created yet ({e})")
