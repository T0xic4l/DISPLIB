import os
import subprocess


for instance in [f for f in os.listdir("Phase1Instances") if f.endswith(".json")]:
    subprocess.run(f"python main.py {instance}")