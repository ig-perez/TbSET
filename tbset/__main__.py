import os
from tbset.app import TbSETPrompt


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    prompt = TbSETPrompt()
    prompt.start()
