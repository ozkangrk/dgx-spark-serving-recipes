import json
d = json.load(open('/home/ozkangu/Desktop/gb10-repo/evidence/c3-draft16-bf16kv-20260902/hermes-c123-v2/C2/B/result.json'))
assert d == {"task": "B", "sum": 15, "count": 5}, d
print("VALID: exact match, keys =", list(d.keys()))
