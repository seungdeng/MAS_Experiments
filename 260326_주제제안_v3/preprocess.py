# import json

# # 파일 불러오기
# with open("./data/GSM8K.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # task_id 추가 (1부터 시작)
# for i, item in enumerate(data, start=1):
#     item["task_id"] = i

# # 다시 저장
# with open("GSM8K_with_task_id.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("완료: task_id 추가됨")

import json

file_path = "./rawdata/MATH.json"  # 원본 파일
output_path = "./data/MATH.json"  # 저장 파일

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# data가 리스트라고 가정
for item in data:
    if "unique_id" in item:
        item["task_id"] = item.pop("unique_id")  # 이름 변경

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("변환 완료")