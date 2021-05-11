import string
import math
import re
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
import pyspark.sql.types as T
from pyspark.sql.types import *
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import NGram, Tokenizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


@udf
def processing (text):
    text = text.lower().strip().translate(string.maketrans(“”,””), string.punctuation)
    stop_words = set(ENGLISH_STOP_WORDS)
    tokens = word_tokenize(text)
    normalize_text = [i for i in tokens if not i in stop_words]
    return normalize_text


def make_ngrams (df, n=1):

    df = df.withColumn('normalized_text', processing(F.col('text')))
    tokenizer = Tokenizer(inputCol="normalized_text", outputCol="tokens")
    tokenized = tokenizer.transform(df).drop('normalized_text')
    
    ngram = NGram(n=n, inputCol="tokens", outputCol="n_gram")
    n_gram_df = ngram.transform(tokenized)
    n_gram_df = n_gram_df.withColumn('n_gram', F.explode('n_gram'))
    n_gram_df = n_gram_df.filter(F.length('n_gram')>2)
    
    return n_gram_df



def chi_square_procedur(n_gram_df, usable_words, min_count=7):
    
    word_freq_each_cat = n_gram_df.groupby('n_gram', 'category').count().sort\
        ('count', ascending=False).fillna(0).select('n_gram', 'category', 'count')
    
    word_freq_each_cat = word_freq_each_cat.filter(F.col('count') > min_count)
    
    cat_word_count = word_freq_each_cat.groupby('category').agg(F.sum('count')\
        .alias('categories_count'))

    category_words_count = cat_word_count.crossJoin( 
        cat_word_count.select(F.sum('categories_count').alias("sum_total_count")))\
        .withColumn("cat_total_prob", F.col("categories_count") / F.col("sum_total_count"))
    
    each_word_total_count = word_freq_each_cat.groupBy('n_gram').sum()\
        .withColumnRenamed('sum(count)', 'total_count').select('n_gram','total_count')
        
    each_word_probability = each_word_total_count.crossJoin\
        (each_word_total_count.select(F.sum('total_count').alias("sum_total_count")))\
        .withColumn("word_total_prob", F.col("total_count")/F.col("sum_total_count"))
    
    word_freq_percentage_each_cat = word_freq_each_cat.select('n_gram','category','count')\
        .withColumn("word_percentage_by_category", F.col("count")/F.sum("count")\
        .over(Window.partitionBy('n_gram'))).drop('count')

    df = word_freq_each_cat.join(usable_words, ['n_gram','category'], how='inner')\
        .join(category_words_count.select('cat_total_prob','category','categories_count'), ['category']).dropDuplicates()

    df = df.join(each_word_probability.select('n_gram','word_total_prob','total_count'), ['n_gram'])\
        .dropDuplicates().join(word_freq_percentage_each_cat, ['n_gram','category']).dropDuplicates()
        
    N = n_gram_df.count()

    @udf
    def approximate_chi_score(cat_total_prob, categories_count, word_total_prob, total_count, count):
        
        E1 = (cat_total_prob * word_total_prob * N)
        F1 = count
        X1 = float((F1-E1)*(F1-E1)/E1)
        
        E2 = (1-word_total_prob) * (cat_total_prob) * N
        F2 = categories_count-F1
        X2 = float((F2-E2)*(F2-E2)/E2)
        
        E3 = (word_total_prob) * (1-cat_total_prob) * N
        F3 = total_count - F1
        X3 = float((F3-E3)*(F3-E3)/E3)
        
        return float (X1+X2+X3)
        
    df = df.withColumn('aprx_chi_scr', approximate_chi_score\
                        (F.col('cat_total_prob'), F.col('categories_count'),\
                        F.col('word_total_prob'), F.col('total_count'), F.col('count'))) 
    
    return df


def extractor (df, min_count, output_path):

    n_gram_df = make_ngrams (df)
    
    n_gram_score = chi_square_procedur(n_gram_df, min_count)
    
    window = Window.partitionBy(n_gram_score['category'])\
            .orderBy(n_gram_score['aprx_chi_scr'].desc())
            
    n_gram_score = n_gram_score.dropDuplicates(['n_gram', 'category'])    
    top_word_df = n_gram_score.select('*', F.rank().over(window).alias('rank'))\
                .filter(F.col('rank')<=1000)
                
    top_word_df = top_word_df.join(categories, on = ['category'], how='left')
    
    top_words = top_word_df.orderBy(F.col('category'), F.col('count').desc()).select('n_gram','category',\
                'count','distinct_user_count','aprx_chi_scr').toPandas()
    
    top_words.to_csv(output_path)
    
    return top_words
    
