import json
from collections import defaultdict

PENALTY_VALUE_DICT = {
    "collisions_pedestrian": 1.0,
    "collisions_vehicle": 0.7,
    "collisions_layout": 0.6,
    "red_light": 0.4,
    "stop_infraction": 0.25,
    "scenario_timeouts": 0.4,
    "yield_emergency_vehicle_infractions": 0.4,
    "min_speed_infractions": 0.4,
}

def compute_penalty_v21(infractions):
    infraction_value = 0.0
    for key, messages in infractions.items():
        if key in PENALTY_VALUE_DICT:
            infraction_value += PENALTY_VALUE_DICT[key] * len(messages)
    return 1.0 / (1.0 + infraction_value)

def recompute_scores(data):
    records = data["_checkpoint"]["records"]

    updated_records = []
    total_score_route = 0
    total_score_penalty = 0
    total_score_composed = 0
    completed_routes = 0
    total_km = 0
    infraction_counts = defaultdict(int)

    for record in records:
        route_length_km = record["meta"]["route_length"] / 1000.0
        total_km += route_length_km

        route_score = record["scores"]["score_route"]
        infraction_penalty = compute_penalty_v21(record["infractions"])
        composed_score = route_score * infraction_penalty

        record["scores"]["score_penalty_v2_1"] = round(infraction_penalty, 6)
        record["scores"]["score_composed_v2_1"] = round(composed_score, 6)

        total_score_route += route_score
        total_score_penalty += infraction_penalty
        total_score_composed += composed_score

        if "Perfect" in record["status"] or "Completed" in record["status"]:
            completed_routes += 1

        # count all infractions for stats per km
        for k, v in record["infractions"].items():
            infraction_counts[k] += len(v)

        updated_records.append(record)

    num_routes = len(records)
    summary = {
        "avg_score_route": round(total_score_route / num_routes, 6),
        "avg_score_penalty_v2_1": round(total_score_penalty / num_routes, 6),
        "avg_score_composed_v2_1": round(total_score_composed / num_routes, 6),
        "success_rate": round(completed_routes / num_routes, 6),
        "infractions_per_km": {
            k: round(v / total_km, 6) for k, v in infraction_counts.items()
        }
    }

    return updated_records, summary

# --- Usage Example ---
if __name__ == "__main__":
    import sys

    input_file = ""  # Replace with your path
    output_file = ""

    with open(input_file, "r") as f:
        data = json.load(f)

    updated_records, summary = recompute_scores(data)

    data["_checkpoint"]["records"] = updated_records
    data["v2.1_summary"] = summary

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print("Recomputed results saved to", output_file)
    print("Summary (v2.1):", json.dumps(summary, indent=2))
