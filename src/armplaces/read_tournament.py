"""Восстановление исторического турнира из текстовых протоколов DE_OLD.

Последняя строка [DE] содержит +, -, >, <: исходы боёв и технических проходов.
Участники перемещаются по _fix-копиям адресных таблиц; нулевой адрес игнорируется.
Возвращается исходный список реальных боёв с повторениями. Отдельная функция
строит проекцию для GrinTour, не изменяя журнал. Ошибки данных не скрываются.
Форматы и исправления таблиц: README.md, раздел «Исторические протоколы и DE_OLD».
"""
from collections import Counter
from math import isfinite
from pathlib import Path

from armplaces.tables.DE_OLD_Loser_fix import DE_OLD_loser_fix as DE_OLD_loser
from armplaces.tables.DE_OLD_Winner_fix import DE_OLD_winner_fix as DE_OLD_winner

RESULT_FILE_SUFFIX = "[DE]"
RESULT_FILE_5_6_SUFFIX = "[DE_5-6]"
WEIGHTS_FILE_SUFFIX = "[ADE]"
WEIGHTS_FILE_5_6_SUFFIX = "[DE_0_5-6]"
SYMBOLS = set("+-<>")


def read_names(lines: list) -> tuple:
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) < 2 or not set(lines[-1]) <= SYMBOLS:
        raise ValueError("Expected names followed by a result word containing + - < >")
    names = lines[:-1]
    if len(set(names)) != len(names):
        raise ValueError("Participant names must be unique")
    return dict(enumerate(names, 1)), list(lines[-1])


def read_weights(lines: list) -> dict:
    values = [float(line.strip().replace(",", ".")) for line in lines if line.strip()]
    if any(not isfinite(value) or value <= 0 for value in values):
        raise ValueError("Weights must be finite and positive")
    return dict(enumerate(values, 1))


# Backward-compatible spelling used by the original project.
read_weigths = read_weights


def read_tournament_files(filepath: Path | str) -> dict:
    directory = Path(filepath)
    if not directory.is_dir():
        raise ValueError(f"Tournament directory does not exist: {directory}")
    files = dict.fromkeys((RESULT_FILE_SUFFIX, RESULT_FILE_5_6_SUFFIX,
                          WEIGHTS_FILE_SUFFIX, WEIGHTS_FILE_5_6_SUFFIX))
    for path in sorted(directory.glob("*.txt")):
        for suffix in files:
            if path.stem.endswith(suffix):
                if files[suffix] is not None:
                    raise ValueError(f"Multiple {suffix} files in {directory}")
                lines = path.read_text(encoding="utf-8-sig").splitlines()
                try:
                    files[suffix] = (read_names(lines) if suffix in
                                     (RESULT_FILE_SUFFIX, RESULT_FILE_5_6_SUFFIX)
                                     else read_weights(lines))
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                break
    if files[RESULT_FILE_SUFFIX] is None:
        raise ValueError(f"Missing {RESULT_FILE_SUFFIX} protocol in {directory}")
    for result_key, weight_key in ((RESULT_FILE_SUFFIX, WEIGHTS_FILE_SUFFIX),
                                   (RESULT_FILE_5_6_SUFFIX, WEIGHTS_FILE_5_6_SUFFIX)):
        weights = files[weight_key]
        if weights is not None:
            if files[result_key] is None or len(weights) != len(files[result_key][0]):
                raise ValueError(f"Names and weights disagree for {weight_key}")
    return files


def drop_simple_cycles(pairs: list) -> list:
    """Graph-only projection: deduplicate, keep majority; retain tied directions."""
    counts = Counter(tuple(pair) for pair in pairs if "" not in pair)
    return [pair for pair, count in counts.items()
            if count >= counts[(pair[1], pair[0])]]


def tournament_recovery(files: dict) -> list:
    """Recover raw (loser, winner) bouts; technical passes are not bouts."""
    if not files.get(RESULT_FILE_SUFFIX):
        raise ValueError("Missing main protocol")
    names, results = files[RESULT_FILE_SUFFIX]
    n = len(names)
    if not 2 <= n <= 32 or set(names) != set(range(1, n + 1)):
        raise ValueError("DE_OLD requires 2..32 consecutively numbered participants")
    if not results or len(results) > 2 * n - 1 or any(x not in SYMBOLS for x in results):
        raise ValueError("Invalid DE_OLD result word")
    max_address = max(max(row[n - 2] for row in DE_OLD_winner),
                      max(row[n - 2] for row in DE_OLD_loser))
    sequence = [0] * max(max_address, 2 * len(results), n)
    sequence[:n] = range(1, n + 1)
    pairs = []
    for i, result in enumerate(results):
        a, b = sequence[2 * i:2 * i + 2]
        winner, loser = (a, b) if result in "+>" else (b, a)
        if not winner or winner == loser or (result in "+-" and not loser):
            raise ValueError(f"Invalid participants in DE_OLD pair {i + 1}: {a}, {b}")
        if result in "+-":
            pairs.append((names[loser], names[winner]))
        for participant, table in ((winner, DE_OLD_winner), (loser, DE_OLD_loser)):
            destination = table[i][n - 2]
            if destination:
                if destination <= 2 * i + 2:
                    raise ValueError(f"Backward DE_OLD transition in pair {i + 1}")
                if sequence[destination - 1]:
                    raise ValueError(f"Conflicting DE_OLD transition in pair {i + 1}")
                sequence[destination - 1] = participant
    extra = files.get(RESULT_FILE_5_6_SUFFIX)
    if extra:
        extra_names, extra_results = extra
        if set(extra_names) != {1, 2} or not set(extra_names.values()) <= set(names.values()):
            raise ValueError("Invalid participants in 5–6 protocol")
        if not extra_results or extra_results[0] not in SYMBOLS:
            raise ValueError("Invalid 5–6 result")
        if extra_results[0] in "+-":
            winner, loser = (1, 2) if extra_results[0] == "+" else (2, 1)
            pairs.append((extra_names[loser], extra_names[winner]))
    return pairs
