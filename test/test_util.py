# encoding=utf-8
from src.util.logger import setlogger
from src.util.yaml_util import loadyaml

try:
    config = loadyaml('../conf/nlptools.yaml')
    logger = setlogger(config)
    logger.info(config['w2v_path'])
except Exception as e:
    logger.error(e)