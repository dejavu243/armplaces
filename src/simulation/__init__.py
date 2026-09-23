"""Публичный API симуляции: Config, генерация поля, сетка и проведение турнира.

Экспортируемые функции не записывают файлы. Серии с экспортом запускаются через
simulation.experiments.run_experiments. Примеры Python API — в README.md.
"""
from .core import Config, build_bracket, elo_probability, generate_field, simulate, validate_result

__all__ = ['Config', 'build_bracket', 'elo_probability', 'generate_field', 'simulate', 'validate_result']
