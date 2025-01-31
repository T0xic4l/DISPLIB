import os
import subprocess


for instance in [f for f in os.listdir("Solved") if f.endswith(".json")]:
    subprocess.run(f"python displib_verify.py Solved/{instance} Solutions/10min_sol_{instance}")

