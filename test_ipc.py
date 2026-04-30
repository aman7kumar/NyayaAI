from backend.models.ipc_classifier import IPCClassifier

# Load model
classifier = IPCClassifier.load()

# Test cases
tests = [
    "Someone stole my phone in market",
    "My husband beats me daily",
    "I was threatened with a knife",
    "My bank account got hacked",
    "Police are not registering FIR"
]

for t in tests:
    print("\nINPUT:", t)
    results = classifier.predict(t)

    for r in results:
        print(f"→ {r['section']} ({r['title']}) | Confidence: {r['confidence']:.2f}")