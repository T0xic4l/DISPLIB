import os
import subprocess


for instance in [f for f in os.listdir("Instances") if f.endswith(".json")]:
    subprocess.run(f"python displib_verify.py Instances/{instance} Solutions/sol_{instance}")