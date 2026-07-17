"""Test portabilità percorsi Windows/Linux senza dipendere dal sistema ospite."""

import os
import tempfile
from pathlib import Path

from path_utils import (normalizza_per_confronto, percorso_da_salvare,
                        percorso_e_portabile, risolvi_percorso)


def esegui_test():
    with tempfile.TemporaryDirectory() as cartella:
        radice = Path(cartella) / "DeepSight"
        archivio = radice / "data" / "archive"
        archivio.mkdir(parents=True)
        file_locale = archivio / "abc.jpg"
        file_locale.write_bytes(b"x")

        assert percorso_da_salvare(file_locale, radice) == "data/archive/abc.jpg"
        assert percorso_e_portabile("data/archive/abc.jpg")
        assert not percorso_e_portabile(vecchio_windows :=
                                        r"C:\\Utenti\\Mario\\DeepSight\\data\\archive\\abc.jpg")
        assert risolvi_percorso("data/archive/abc.jpg", radice) == str(file_locale.resolve())

        assert percorso_da_salvare(vecchio_windows, radice) == "data/archive/abc.jpg"
        assert risolvi_percorso(vecchio_windows, radice) == str(file_locale.resolve())
        assert percorso_da_salvare(
            r"D:\\DEEPSIGHT\\DATA\\ARCHIVE\\abc.jpg", radice
        ) == "data/archive/abc.jpg"
        vecchio_linux = "/home/mario/DeepSight/data/archive/abc.jpg"
        assert percorso_da_salvare(vecchio_linux, radice) == "data/archive/abc.jpg"
        assert risolvi_percorso(vecchio_linux, radice) == str(file_locale.resolve())

        esterno = Path(cartella) / "esterno.jpg"
        assert percorso_da_salvare(esterno, radice) == str(esterno.resolve())
        assert normalizza_per_confronto("data/archive/abc.jpg", radice) == os.path.normcase(
            str(file_locale.resolve())
        )
    print("test_path_utils: OK")


if __name__ == "__main__":
    esegui_test()
