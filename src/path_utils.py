"""Percorsi persistiti portabili tra Windows e Linux.

Nel database i file gestiti da DeepSight sono salvati relativamente alla radice
del progetto (es. ``data/archive/abc.jpg``). Al confine con filesystem/UI vengono
risolti in percorsi assoluti. I vecchi percorsi assoluti restano leggibili e, se
contengono la cartella ``data``, possono essere rilocati su un altro sistema.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import config


_CARTELLE_GESTITE = {"archive", "frames", "faces", "thumbnails", "db", "quarantena", "log"}


def _radice(base_dir=None) -> Path:
    return Path(base_dir or config.DIR_BASE).resolve()


def _relativo_data_legacy(valore: str) -> str | None:
    """Estrae ``data/...`` anche da un percorso assoluto dell'altro sistema."""
    parti = [p for p in valore.replace("\\", "/").split("/") if p]
    indici = [i for i, parte in enumerate(parti) if parte.casefold() == "data"]
    for indice in reversed(indici):
        coda = parti[indice:]
        if len(coda) >= 2 and coda[1].casefold() in _CARTELLE_GESTITE:
            coda[0] = "data"
            coda[1] = coda[1].casefold()
            return "/".join(coda)
    return None


def percorso_da_salvare(percorso, base_dir=None) -> str | None:
    """Converte un percorso gestito in testo relativo, usando sempre ``/``.

    I percorsi esterni alla radice vengono conservati assoluti: DeepSight non li
    crea normalmente, ma non li modifica alla cieca durante una migrazione.
    """
    if percorso is None:
        return None
    valore = os.fspath(percorso).strip()
    if not valore:
        return valore

    radice = _radice(base_dir)
    nativo = Path(valore)
    windows_assoluto = PureWindowsPath(valore).is_absolute()
    posix_assoluto = PurePosixPath(valore).is_absolute()

    if not nativo.is_absolute() and not windows_assoluto and not posix_assoluto:
        normalizzato = valore.replace("\\", "/")
        candidato = (radice / normalizzato).resolve()
        try:
            return candidato.relative_to(radice).as_posix()
        except ValueError:
            return normalizzato

    if nativo.is_absolute():
        try:
            return nativo.resolve().relative_to(radice).as_posix()
        except ValueError:
            pass

    relativo_legacy = _relativo_data_legacy(valore)
    if relativo_legacy:
        return relativo_legacy
    return str(nativo.resolve()) if nativo.is_absolute() else valore


def risolvi_percorso(percorso, base_dir=None) -> str | None:
    """Restituisce il percorso filesystem assoluto corrispondente al valore DB."""
    if percorso is None:
        return None
    valore = os.fspath(percorso).strip()
    if not valore:
        return valore

    radice = _radice(base_dir)
    nativo = Path(valore)
    if nativo.is_absolute():
        # Un assoluto legacy ancora valido va rispettato. Se il progetto è stato
        # spostato, il vecchio path non esiste più e si riloca tramite ``data/...``.
        if nativo.exists():
            return str(nativo)
        relativo_legacy = _relativo_data_legacy(valore)
        return str(radice / relativo_legacy) if relativo_legacy else str(nativo)

    if PureWindowsPath(valore).is_absolute() or PurePosixPath(valore).is_absolute():
        relativo_legacy = _relativo_data_legacy(valore)
        return str(radice / relativo_legacy) if relativo_legacy else valore

    return str((radice / valore.replace("\\", "/")).resolve())


def normalizza_per_confronto(percorso, base_dir=None) -> str:
    """Forma stabile per confrontare un percorso DB con un file sul disco."""
    risolto = risolvi_percorso(percorso, base_dir=base_dir) or ""
    return os.path.normcase(str(Path(risolto).resolve()))


def percorso_e_portabile(percorso) -> bool:
    """True se il valore DB è relativo e non può uscire dalla radice."""
    if percorso is None:
        return True
    valore = os.fspath(percorso).strip()
    if not valore:
        return True
    if (Path(valore).is_absolute() or PureWindowsPath(valore).is_absolute()
            or PurePosixPath(valore).is_absolute()):
        return False
    parti = [p for p in valore.replace("\\", "/").split("/") if p not in ("", ".")]
    return ".." not in parti
