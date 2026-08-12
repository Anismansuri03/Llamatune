import subprocess

LLAMA_SERVER = "bin/llama-server.exe"

def launch_server(model_path: str, ngl: int, port: int = 8080):
    cmd = [LLAMA_SERVER, "-m", model_path, "-ngl", str(ngl), "--port", str(port)]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd)