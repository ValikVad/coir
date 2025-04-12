import argparse
import logging
import configparser
import sys
import os

# Добавляем текущую директорию в системный путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coir.data_loader import get_tasks, load_from_csv
from coir.evaluation import COIR
from coir.models import YourCustomDEModel, APIDEModel

def setup_logging(verbose):
    """
    Настраивает логирование.

    Args:
        verbose (bool): Включить подробный вывод.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def read_config(config_path):
    """
    Читает конфигурационный файл.

    Args:
        config_path (str): Путь к конфигурационному файлу.

    Returns:
        configparser.ConfigParser: Конфигурационный объект.
    """
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        logging.info(f"Конфигурационный файл {config_path} успешно прочитан")
        return config
    except Exception as e:
        logging.error(f"Ошибка при чтении конфигурационного файла: {e}")
        raise

def main():
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Бенчмарк для сравнения моделей машинного обучения.")
    parser.add_argument('--data_local', type=bool, default=False, help='Флаг для использования локальных данных')
    parser.add_argument('--data', type=list, default=["codesearchnet"], help='Путь к CSV файлу с задачами или название датасета с HF')
    parser.add_argument('--config', type=str, default='config.ini', help='Путь к конфигурационному файлу')
    parser.add_argument('--verbose', action='store_true', help='Включить подробный вывод')
    parser.add_argument('--use_api', action='store_true', help='Использовать API для инференса модели')
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Чтение конфигурационного файла
    config = read_config(args.config)

    # Загрузка задач
    if args.data_local:
        tasks = load_from_csv(args.data)
    else:
        tasks = get_tasks(args.data)
    if not tasks:
        logging.error("Не удалось загрузить задачи. Проверьте путь к файлу.")
        return

    # Инициализация модели
    if args.use_api:
        model_config = config['api_model']
        model = APIDEModel(model_config)  # Предполагается, что у вас есть такой класс
    else:
        model_config = config['model']
        model = YourCustomDEModel(model_config)

    # Инициализация и оценка модели
    evaluator = COIR(model)
    results = evaluator.evaluate(tasks)

    # Вывод результатов
    for task_id, result in results.items():
        if 'error' in result:
            logging.error(f"Ошибка при оценке задачи {task_id}: {result['error']}")
        else:
            logging.info(f"Результаты оценки задачи {task_id}: {result}")

if __name__ == "__main__":
    main()