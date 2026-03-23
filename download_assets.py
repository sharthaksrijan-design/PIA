import os
import sys
import subprocess
import zipfile

def download_glove():
    GLOVE_PATH = 'glove.6B.100d.txt'
    if not os.path.exists(GLOVE_PATH):
        print("Downloading GloVe 100d...")
        subprocess.run(['wget', '-q', '-O', 'glove.6B.zip',
            'https://nlp.stanford.edu/data/glove.6B.zip'], check=True)
        print("Unzipping GloVe...")
        with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
            zip_ref.extract('glove.6B.100d.txt', '.')
        os.remove('glove.6B.zip')
    print(f"GloVe 100d ready: {os.path.getsize(GLOVE_PATH):,} bytes")

def install_deps():
    print("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets', 'nltk', 'numpy', 'scipy', '-q'], check=True)
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

if __name__ == "__main__":
    install_deps()
    download_glove()
