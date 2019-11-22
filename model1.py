# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

from loader1 import load_spark_df, load_pandas_df
import pyspark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics


def get_als_model_rmse(df, rank):
    train, test = df.randomSplit([0.9, 0.1], seed=1)
    als = ALS(
        maxIter=5,
        regParam=0.09,
        rank=rank,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True)
    
    model = als.fit(train)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction")
    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)
    print(f'RMSE is {rmse}')
    return (predictions, rmse, model)


def calculate_coverage(model):
    user_recos = model.recommendForAllUsers(numItems=10)
    recos_list = user_recos.select('recommendations').collect()
    recos_list = [el for el in recos_list]
    recos_list = [x for b in recos_list for x in b]
    recos_list = [item for sublist in recos_list for item in sublist]
    movie_list = [row['movieId'] for row in recos_list]
    movie_set = list(set(movie_list))
    return movie_set


def get_best_rank(df):
    #based on rmse
    rmse_dict = {}
    coverage_dict = {}
    for rank in [1, 2, 4, 8, 16, 32, 64, 128]:
#    for rank in [64, 128]:
        print(f'Rank is {rank}')
        _, rmse, model = get_als_model_rmse(df, rank)
        coverage = calculate_coverage(model)
        rmse_dict[rank] = rmse
        coverage_dict[rank] = coverage
    return rmse_dict, coverage_dict

def get_rank_report(df):
    rank = 64
    predictions, rmse = get_als_model_rmse(df, rank)
    metrics = RegressionMetrics(predictions)
    print("RMSE = %s" % metrics.rootMeanSquaredError)
    


if __name__ == '__main__':
    dir_name = 'ml-latest-small'
    ratings_spark_df = load_spark_df(dir_name, 'ratings', use_cache=True)
    rmse_dict = get_best_rank(ratings_spark_df)
    #get_rank_report(ratings_spark_df)
#    print("RMSE=" + str(rmse))
#    predictions.show()
