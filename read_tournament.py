"""
Read tournament from files

1. Вычитываем N спортсменов из файла *_[DE].txt (спортсмены указаны в порядке после жеребьевки);
2. Вычитываем результаты поединков Res (последняя строка в файле *_[DE].txt);
3. Вычитываем веса спортсменов из файла *_[ADE].txt (для файла спортсменов из *_[DE].txt)
4. Если длина слова Res с результатами поединков M=2*(N-1), то в категории нет суперфинала, если M=2*(N-1)+1,
то в категории был суперфинал. Длина слова M будет использоваться в дальнейшем;
5. Задаем генеральную последовательность (линейный массив GS) длины (2*M+N);
6. Записываем номера спортсменов (помним соответствие между номеров и именем) в генеральную последовательность
с 1-го элемента;
7. Запускаем цикл по GS, двигаясь по парам, т.е. с шагом в 2 элемента последовательности (i – номер итерации / пары).
Если в i-й ячейке Res записан ‘+’, то номер первого из пары перемещается в ячейку GS c номером DE_OLD_Winner[i][N+1],
а проигравшего в DE_OLD_Loser[i][N+1], если ‘-’, то наоборот. Проходим все пары  до i = M.
В процессе формируем массив пар для построения графа без перестановки порядка в паре, если ‘+’
и с перестановкой порядка, если ‘-’. Если в Res записан ‘>’, то пара в массив для графа не записывается,
так как это техническая победа, но перемещение в GS равносильно ‘+’, аналогично для ‘<’ и ‘-’;
8. В процессе движения записываем следующие параметры для каждого спортсмена: кол-во побед Win; номер того,
кто победил спортсмена Num_Los;9. Вычитываем пару поединка за 5-6 места из *_[DE_5-6].txt и Res_5-6, из которого
нужен только первый символ. Согласно ‘+’ или ‘–‘ записываем пару в массив для построения графа;
10. Если требуется, то вычитываем веса спортсменов, участвующих в поединке за 5-6 места из _[DE_0_5-6].txt;
"""

import logging
from pathlib import Path
from glob import glob
from typing import Union

from data.DE_OLD_Winner import DE_OLD_winner
from data.DE_OLD_Loser import DE_OLD_loser

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)

RESULT_FILE_SUFFIX = "[DE]"
RESULT_FILE_5_6_SUFFIX = "[DE_5-6]"
WEIGHTS_FILE_SUFFIX = "[ADE]"
WEIGHTS_FILE_5_6_SUFFIX = "[DE_0_5-6]"

CURRENT_DIR = Path(__file__).parent.resolve()


def read_names(lines: list) -> tuple:
    names = {}
    results = None
    for i, name in enumerate(lines):
        if "+" not in name and "-" not in name:
            names[i] = name.strip()
        else:
            results = list(name.strip())
    return names, results


def read_weigths(lines: list) -> dict:
    weights = {}
    for i, weight in enumerate(lines):
        weights[i] = float(weight.strip().replace(",", "."))
    return weights

def make_pairs(names: dict, results: list) -> dict:
    pass


def read_results_file(filepath: Union[Path, str]) -> dict:
    filelist = glob(filepath + "*.txt")
    logger.info("filelist: %s", filelist)
    files = {}
    for filename in filelist:
        try:
            file = open(filename, "r", encoding="utf-8")
            lines = file.readlines()
            if RESULT_FILE_SUFFIX in filename:
                names, results = read_names(lines)
                files[RESULT_FILE_SUFFIX] = [names, results]
                logger.debug(f"{names=}")
                logger.debug(f"{results=}")
            elif WEIGHTS_FILE_SUFFIX in filename:
                weights = read_weigths(lines)
                files[RESULT_FILE_SUFFIX] = weights
                logger.debug(f"{weights=}")
            elif RESULT_FILE_5_6_SUFFIX in filename:
                names, results = read_names(lines)
                files[RESULT_FILE_5_6_SUFFIX] = [names, results]
                logger.debug(f"{names=}")
                logger.debug(f"{results=}")
            elif WEIGHTS_FILE_5_6_SUFFIX in filename:
                weights = read_weigths(lines)
                files[WEIGHTS_FILE_5_6_SUFFIX] = weights
                logger.debug(f"{weights=}")
            else:
                logger.warning("File %s not mathing with tournament files", filename)
        except Exception as error:
            logger.error("File %s doesn't read, check existence or encoding \n%s", filename, error)

    return files


read_results_file('./data/left_hand_75kg/')
