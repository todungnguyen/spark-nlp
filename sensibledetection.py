# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import time
import pandas as pd
import dataprofiler as dp
import sparknlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from sparknlp.pretrained import PretrainedPipeline

class SensibleDetection:    
    def __init__(self):
        start = time.time()
        
        self._spark = SparkSession.builder \
                        .appName("Spark NLP")\
                        .master("local[4]")\
                        .config("spark.driver.memory","16G")\
                        .config("spark.driver.maxResultSize", "0") \
                        .config("spark.kryoserializer.buffer.max", "2000M")\
                        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.2")\
                        .getOrCreate()
        self._pipeline = PretrainedPipeline.from_disk('entity_recognizer_lg_fr')
        self._unknown_cols = []
        self._df_result = pd.DataFrame()
        
        stop = time.time()
        print("Initialize time: %.2f" % (stop - start), "s")
    
    def stop(self):
        self._spark.stop()
        
    def _get_structured_results(self, results):
        columns = []
        predictions = []
        for col_report in results['data_stats']:
            columns.append(col_report['column_name'])
            predictions.append(col_report['data_label'])
        self._df_result = pd.DataFrame({'Column' : columns, 'Prediction' : predictions})
    
    """
    def _get_entity(self, column):
        df_spark = self._spark.read.format("csv").option("header","true").option("inferSchema","true").load(self._path)
        data = df_spark.select(column).toDF("text")
        annotations = self._pipeline.transform(data)
        list_ner = annotations.selectExpr("ner.result AS ner").collect()
        return self._get_entity_from_list(list_ner)
    
    def _get_entity_from_list(self, list_ner):
        result = []
        for ner in list_ner:
            count_entity = {'PER': 0, 'LOC': 0, 'MISC': 0, 'ORG': 0}
            for i in range(len(ner.ner)):
                for key in count_entity.keys():
                    if key in ner.ner[i]:
                        count_entity[key] += 1
            max_key = max(count_entity, key = count_entity.get)
            if count_entity[max_key]/len(ner.ner) < 2/3:
                max_key = "UNKNOWN"
            result.append(max_key)
        return max(result, key = result.count)
    """
    
    def _get_entity(self, column):
        df_spark = self._spark.read.option("header","true").csv(self._path)
        data = df_spark.select(column).toDF("text")
        annotations = self._pipeline.transform(data)
        result = annotations.select(F.col("text"), F.explode("ner.result").alias("entity"))

        result = result.withColumn('ent', F.when(result['entity'] != "O", F.split(result['entity'], '-').getItem(1)) \
                                               .otherwise(result['entity'])) \
                        .drop("entity") \
                        .groupby("text", "ent") \
                        .agg(F.count("ent").alias("count"))

        w = Window.partitionBy("text").orderBy(F.col("count").desc())

        result = result.withColumn("row",F.row_number().over(w)) \
                        .filter(F.col("row") == 1) \
                        .drop("row").orderBy("text") \
                        .groupby("ent").count() \
                        .orderBy(F.col("count").desc())
    
        return result.first()[0]
    
    def _run_dp(self):
        start = time.time()
        
        data = dp.Data(self._path)
        profiler = dp.Profiler(data)
        results = profiler.report(report_options={'output_format':'compact'})
        self._get_structured_results(results)
        
        for index, row in self._df_result.iterrows():
            if row['Prediction'] == 'UNKNOWN':
                self._unknown_cols.append(row['Column'])
                
        stop = time.time()
        print("Data Profiler run time: %.2f" % (stop - start), "s")
    
    def _run_snlp(self):
        start = time.time()
        
        for col in self._unknown_cols:
            self._df_result.at[self._df_result['Column'] == col, 'Prediction'] = self._get_entity(col)
            
        stop = time.time()
        print("Spark NLP run time: %.2f" % (stop - start), "s")

    def run(self, path):
        self._path = path
        
        self._run_dp()
        print(self._df_result)
        
        self._run_snlp()
        
        self.stop()
        return self._df_result
# -


