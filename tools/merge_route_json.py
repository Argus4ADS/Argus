import json
import glob
import argparse
import os

def merge_route_json(folder_path):
    file_paths = glob.glob(f'{folder_path}/*.json')
    merged_records = []
    driving_score = []
    route_complete = []
    
    total_collisions = 0
    total_stuck = 0
    total_stop_violations = 0
    total_route_length = 0.0
    success_num = 0
    for file_path in file_paths:
        if 'merged.json' in file_path: continue
        with open(file_path) as file:
            data = json.load(file)
            records = data['_checkpoint']['records']
            for rd in records:
                rd.pop('index')
                merged_records.append(rd)
                driving_score.append(rd['scores']['score_composed'])
                route_complete.append(rd['scores']['score_route'])
                
                collisions = (
                    rd['infractions']['collisions_layout'] +
                    rd['infractions']['collisions_pedestrian'] +
                    rd['infractions']['collisions_vehicle']
                )
                total_collisions += len(collisions)
                total_stuck += len(rd['infractions']['vehicle_blocked'])
                total_stop_violations += len(rd['infractions']['stop_infraction'])
                total_route_length += rd['meta']['route_length'] * rd['scores']['score_route'] /100
                
                if rd['status']=='Completed' or rd['status']=='Perfect':
                    success_flag = True
                    for k,v in rd['infractions'].items():
                        if len(v)>0 and k != 'min_speed_infractions':
                            success_flag = False
                            break
                    if success_flag:
                        success_num += 1
                        print(rd['route_id'])
                
    if len(merged_records) != 220:
        print(f"-----------------------Warning: there are {len(merged_records)} routes in your json, which does not equal to 220. All metrics (Driving Score, Success Rate, Ability) are inaccurate!!!")
    merged_records = sorted(merged_records, key=lambda d: d['route_id'], reverse=True)
    _checkpoint = {
        "records": merged_records
    }

    total_km = total_route_length / 1000.0
    merged_data = {
        "_checkpoint": _checkpoint,
        "driving score": sum(driving_score) / 220,
        "success rate": success_num / 220,
        "route complete": sum(route_complete) / 220,
        "total_collisions": total_collisions,
        "total_stuck": total_stuck,
        "total_stop_sign_violations": total_stop_violations,
        "total_route_length_km": total_km,
        "collisions_per_km": total_collisions / total_km,
        "stuck_per_km": total_stuck / total_km,
        "stop_violations_per_km": total_stop_violations / total_km,
        "eval num": len(driving_score),
    }

    with open(os.path.join(folder_path, 'merged.json'), 'w') as file:
        json.dump(merged_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='old foo help')
    args = parser.parse_args()
    merge_route_json(args.folder)
