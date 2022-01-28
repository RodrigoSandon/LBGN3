from preprocess import preprocess
from pathlib import Path

if __name__ == "__main__":
    from tkinter.filedialog import askopenfilename

    filename = askopenfilename()
    preprocess(Path(filename))
