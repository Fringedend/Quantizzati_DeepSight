"""Funzioni pure di supporto per ricerca, paginazione ed export della Galleria."""

from __future__ import annotations

import datetime as dt
import os
import re
import unicodedata
import zipfile


_DATA_ITALIANA = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


def _normalizza_testo(valore: object) -> str:
    testo = unicodedata.normalize("NFKD", str(valore or ""))
    return "".join(c for c in testo if not unicodedata.combining(c)).casefold()


def _normalizza_termine_ricerca(termine: str) -> str:
    """Converte anche le date italiane in ISO, come sono salvate nel DB."""
    termine = _normalizza_testo(termine).strip()
    corrispondenza = _DATA_ITALIANA.match(termine)
    if not corrispondenza:
        return termine
    giorno, mese, anno = map(int, corrispondenza.groups())
    try:
        return dt.date(anno, mese, giorno).isoformat()
    except ValueError:
        return termine


def filtra_e_ordina_elementi(elementi, testo="", tipo="Tutti", ordinamento="Più recenti"):
    """Filtra senza AI per nome, luogo e data e applica l'ordinamento scelto."""
    termini = [_normalizza_termine_ricerca(t) for t in testo.split() if t.strip()]
    tipo_voluto = {"Immagini": "image", "Video": "video"}.get(tipo)

    filtrati = []
    for elemento in elementi:
        if tipo_voluto and elemento.get("media_type") != tipo_voluto:
            continue
        contenuto = " ".join(
            _normalizza_testo(elemento.get(campo))
            for campo in ("filename", "location_name", "creation_date")
        )
        if termini and not all(termine in contenuto for termine in termini):
            continue
        filtrati.append(elemento)

    if ordinamento == "Nome (A-Z)":
        filtrati.sort(key=lambda e: _normalizza_testo(e.get("filename")))
    else:
        filtrati.sort(
            key=lambda e: (e.get("creation_date") or "", e.get("id") or 0),
            reverse=(ordinamento == "Più recenti"),
        )
    return filtrati


def pagine_compatte(pagina_corrente: int, totale_pagine: int):
    """Pagine numeriche da mostrare; ``None`` rappresenta i puntini di sospensione."""
    totale_pagine = max(1, int(totale_pagine))
    pagina_corrente = min(max(1, int(pagina_corrente)), totale_pagine)
    if totale_pagine <= 7:
        return list(range(1, totale_pagine + 1))

    inizio = max(2, pagina_corrente - 1)
    fine = min(totale_pagine - 1, pagina_corrente + 1)
    if pagina_corrente <= 3:
        fine = 4
    elif pagina_corrente >= totale_pagine - 2:
        inizio = totale_pagine - 3

    risultato = [1]
    if inizio > 2:
        risultato.append(None)
    risultato.extend(range(inizio, fine + 1))
    if fine < totale_pagine - 1:
        risultato.append(None)
    risultato.append(totale_pagine)
    return risultato


def crea_zip_originali(elementi, percorso_zip: str):
    """Crea uno ZIP e ritorna (inclusi, mancanti), senza alterare gli originali."""
    inclusi = []
    mancanti = []
    nomi_usati = set()

    with zipfile.ZipFile(percorso_zip, "w", compression=zipfile.ZIP_DEFLATED) as archivio:
        for elemento in elementi:
            percorso = elemento.get("file_path")
            if not percorso or not os.path.isfile(percorso):
                mancanti.append(elemento.get("filename") or str(elemento.get("id")))
                continue

            nome = os.path.basename(elemento.get("filename") or percorso)
            radice, estensione = os.path.splitext(nome)
            candidato = nome
            progressivo = 2
            while candidato.casefold() in nomi_usati:
                candidato = f"{radice} ({progressivo}){estensione}"
                progressivo += 1
            nomi_usati.add(candidato.casefold())
            archivio.write(percorso, arcname=candidato)
            inclusi.append(candidato)

    return inclusi, mancanti
