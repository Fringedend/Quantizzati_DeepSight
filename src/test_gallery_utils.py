"""Test leggeri per ricerca, paginazione ed export della Galleria."""

import os
import tempfile
import zipfile

import gallery_utils


def esegui_test():
    elementi = [
        {
            "id": 1,
            "filename": "Tramonto.jpg",
            "location_name": "Città di Torino",
            "creation_date": "2026-07-16T12:30:00",
            "media_type": "image",
        },
        {
            "id": 2,
            "filename": "animazione.gif",
            "location_name": "Roma",
            "creation_date": "2025-01-03T09:00:00",
            "media_type": "video",
        },
    ]

    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi, "torino")] == [1]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi, "citta")] == [1]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi, "16/07/2026")] == [1]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi, "roma 2025")] == [2]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi, tipo="Video")] == [2]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(elementi)] == [1, 2]
    assert [e["id"] for e in gallery_utils.filtra_e_ordina_elementi(
        elementi, ordinamento="Meno recenti"
    )] == [2, 1]

    assert gallery_utils.pagine_compatte(1, 3) == [1, 2, 3]
    assert gallery_utils.pagine_compatte(1, 20) == [1, 2, 3, 4, None, 20]
    assert gallery_utils.pagine_compatte(10, 20) == [1, None, 9, 10, 11, None, 20]
    assert gallery_utils.pagine_compatte(20, 20) == [1, None, 17, 18, 19, 20]

    with tempfile.TemporaryDirectory() as cartella:
        primo = os.path.join(cartella, "a.jpg")
        secondo = os.path.join(cartella, "b.jpg")
        with open(primo, "wb") as f:
            f.write(b"a")
        with open(secondo, "wb") as f:
            f.write(b"b")
        percorso_zip = os.path.join(cartella, "selezione.zip")
        inclusi, mancanti = gallery_utils.crea_zip_originali(
            [
                {"id": 1, "filename": "foto.jpg", "file_path": primo},
                {"id": 2, "filename": "foto.jpg", "file_path": secondo},
                {"id": 3, "filename": "manca.gif", "file_path": os.path.join(cartella, "no.gif")},
            ],
            percorso_zip,
        )
        assert inclusi == ["foto.jpg", "foto (2).jpg"]
        assert mancanti == ["manca.gif"]
        with zipfile.ZipFile(percorso_zip) as archivio:
            assert archivio.namelist() == inclusi

    print("test_gallery_utils: OK")


if __name__ == "__main__":
    esegui_test()
