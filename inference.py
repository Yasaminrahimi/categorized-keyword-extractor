import math
import hazm
import re
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import *
from pyspark.ml.feature import NGram, Tokenizer

def processing (text):
