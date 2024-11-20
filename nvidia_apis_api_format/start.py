import subprocess

# Flask API'yi başlat

#subprocess.Popen(["python", "nvidia_apis.py"])
#subprocess.Popen(["python", "neva.py"])
#subprocess.Popen(["python", "vila.py"])
subprocess.Popen(["python", "nvidia_apis.py"])

# Gradio arayüzünü başlat
subprocess.Popen(["python", "interface.py"])
