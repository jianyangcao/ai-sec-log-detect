import pandas as pd
from pathlib import Path

#create folder if it doesn't exist
Path("data").mkdir(exist_ok=True)

#create sample data
data = {
    "timestamp": [
        "2025-10-20 10:01:05",
        "2025-10-20 10:02:10",
        "2025-10-20 10:02:45",
        "2025-10-20 10:03:00",
        "2025-10-20 10:05:30",
        "2025-10-20 10:07:22",
    ],
    "user": ["alice", "bob", "alice", "root", "root", "charlie"],
    "event": ["login", "login", "download", "delete", "logout", "error"],
    "status": ["success", "success", "success", "warning", "success", "error"],
    "ip": [
        "192.168.0.1",
        "192.168.0.2",
        "192.168.0.1",
        "192.168.0.10",
        "192.168.0.10",
        "192.168.0.3",
    ],
}

#convert to dataframe
df=pd.DataFrame(data)

#save to csv
csv_path=Path('data/day_8.csv')
df.to_csv(csv_path,index=False)
print(f"Sample data saved to {csv_path}")