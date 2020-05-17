import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession

import pyspark.sql.functions as f 
from datetime import datetime
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
from pyspark.sql.functions import lower, col
from pyspark.sql import types as t
from pyspark.sql.types import IntegerType

from itertools import chain
from pyspark.sql.functions import col, create_map, lit

import numpy as np

import statsmodels.api as sm

from pyspark.sql.functions import broadcast

from pyspark.sql.functions import *

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    pv = spark.read.csv('hdfs:///tmp/bdm/nyc_parking_violation/', header = True,inferSchema = True)
    #select columns needed for violation counts and joining
    pv = pv.select('Issue Date', 'Violation County', 'Street Name', 'House Number')
    #turn issue date into Date, formatted as a datetime column
    pv = pv.withColumn('Date', from_unixtime(unix_timestamp('Issue Date', 'MM/dd/yyyy')))
    #extract year from date and create new column
    pv = pv.withColumn('Year',f.year(pv['Date']))
    #filter the years we want
    pv = pv.filter(pv["Year"] >= (2015)) \
       .filter(pv["Year"] <= (2019))
    #drop na's and lower case the street name column
    pv = pv.na.drop()
    pv = pv.withColumn('street name',f.lower(pv['Street Name']))
    
    #create a dictionary of possible ways counties are coded to align with the borocode in the centerlines data, to prepare for the join
    borough_dict = {'NY':1, 'MAN':1, 'MH':1, 'NEWY':1, 'NEW':1, 'Y':1, "NY":1,
                'BX':2, 'BRONX':2,
                'K':3, 'BK':3, 'KING':3, 'KINGS':3,
                'Q':4, 'QN':4, 'QNS':4, 'QU':4, 'QUEEN':4,
                'R':5, 'RICHMOND':5, 'ST':5}
    #map the dictionary keys and values to the county vals in the violations df
    mapping_expr = create_map([lit(x) for x in chain(*borough_dict.items())])
    pv = pv.withColumn("BOROCODE", mapping_expr.getItem(col("Violation County")))
    #replace hyphen and remove the subsequent blank spaces, creating a column HN_int (as it will be turned into an int later on
    pv = pv.withColumn("HN_int",(f.regexp_replace("House Number", "-", "")))
    pv = pv.withColumn("HN_int",regexp_replace(col("HN_int"), " ", ""))
    #replace letters from alphanumeric house numbers, so i can still keep slect houses and then cast all house numbers as ints
    pv = pv.withColumn("HN_int", f.regexp_replace(f.col("HN_int"), "[ABCDEFGHIJKLMNOPQRSTUVWXYZ]", ""))
    pv = pv.withColumn("HN_int", pv["HN_int"].cast(IntegerType()))
    pv = pv.na.drop()
    pv = pv.select('Year','BOROCODE', 'street name', 'HN_int')
    #groupby geospatial features and then pivot by the select years 
    #so that I can count the number of violations per House Number per Year (based on boro and street) so it doesn't count across boro's or streets
    #pivot allows the distinct Year values to become their own individual column
    pv = pv.groupBy('BOROCODE', 'street name', 'HN_int').pivot("Year", [2015, 2016, 2017, 2018, 2019]).count()
    pv = pv.na.fill(0)
    
    
    df_centerline = spark.read.csv('hdfs:///tmp/bdm/nyc_cscl.csv', header = True, inferSchema = True)
    df_centerline = df_centerline.select('PHYSICALID', 'ST_LABEL','FULL_STREE', 'BOROCODE', 'L_LOW_HN', 'L_HIGH_HN', 'R_LOW_HN', 'R_HIGH_HN')
    
    #replace hyphen and remove the subsequent blank spaces, creating a column HN_int (as it will be turned into an int later on
    #inspection into sample data showed no letters needed to replaced.
    #same replace method used here so that the numbers are treated the same in both dataframes for the join
    #perform on L and R low and high (evens and odds)
    df_centerline = df_centerline.withColumn("L_LOW_int",(f.regexp_replace("L_LOW_HN", "-", "")))
    df_centerline = df_centerline.withColumn("L_LOW_int",regexp_replace(col("L_LOW_int"), " ", ""))
    df_centerline = df_centerline.withColumn("L_LOW_int", df_centerline["L_LOW_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("L_HIGH_int",(f.regexp_replace("L_HIGH_HN", "-", "")))
    df_centerline = df_centerline.withColumn("L_HIGH_int",regexp_replace(col("L_HIGH_int"), " ", ""))
    df_centerline = df_centerline.withColumn("L_HIGH_int", df_centerline["L_HIGH_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("R_LOW_int",(f.regexp_replace("R_LOW_HN", "-", "")))
    df_centerline = df_centerline.withColumn("R_LOW_int",regexp_replace(col("R_LOW_int"), " ", ""))
    df_centerline = df_centerline.withColumn("R_LOW_int", df_centerline["R_LOW_int"].cast(IntegerType()))
    df_centerline = df_centerline.withColumn("R_HIGH_int",(f.regexp_replace("R_HIGH_HN", "-", "")))
    df_centerline = df_centerline.withColumn("R_HIGH_int",regexp_replace(col("R_HIGH_int"), " ", ""))
    df_centerline = df_centerline.withColumn("R_HIGH_int", df_centerline["R_HIGH_int"].cast(IntegerType()))
    
    #select columns we need and lowercase the street name references for comparison and joining
    df_centerline = df_centerline.select('PHYSICALID', 'ST_LABEL', 'FULL_STREE', 'BOROCODE', 
                                     'L_LOW_int', 'L_HIGH_int', 'R_LOW_int', 'R_HIGH_int')
    df_centerline = df_centerline.withColumn('ST_LABEL', lower(col('ST_LABEL'))).withColumn('FULL_STREE', lower(col('FULL_STREE')))
    
    
    #df with just fullstreet
    full_stree = df_centerline.select('PHYSICALID', 'FULL_STREE', 'BOROCODE', 
                                     'L_LOW_int', 'L_HIGH_int', 'R_LOW_int', 'R_HIGH_int')
    #df with just st_label                                 
    st_label = df_centerline.select('PHYSICALID', 'ST_LABEL','BOROCODE', 
                                     'L_LOW_int', 'L_HIGH_int', 'R_LOW_int', 'R_HIGH_int')
                                     
    
    #create a new df that's the union, looking for distinct vals 
    #this allows the same phys id to exist, if they deviate in other columns (most likely being the 2 street references)
    centerline = st_label.union(full_stree).distinct()
    
    
    #create a df that joins violations with centerlines
    #exact match on both borocode and street name/label (no more or condition needed as there's only one column)
    #%2 condition is to check the remainder, if the remainder isn't 0, compare HN against the odds, if remainder is 0, compare against the evens
    result_df = pv.join(broadcast(centerline),(pv["BOROCODE"]==centerline["BOROCODE"]) & 
                          ((pv["street name"] == centerline['ST_LABEL'])) &
                          (((pv['HN_int']%2!=0) & (pv['HN_int'] >= centerline['L_LOW_int']) & (pv['HN_int'] <= centerline['L_HIGH_int'])) |
                          ((pv['HN_int']%2==0) & (pv['HN_int'] >= centerline['R_LOW_int']) & (pv['HN_int'] <= centerline['R_HIGH_int']))))
    
    #select only the cols we need, order by phys id, group by physid, and sum the counts for the respective years
    result_df = result_df.select('PHYSICALID', '2015', '2016', '2017', '2018', '2019')
    result_df = result_df.orderBy('PHYSICALID')
    result_df =result_df.groupBy('PHYSICALID').agg({'2015' : 'sum', '2016':'sum', '2017':'sum', '2018':'sum', '2019':'sum'})
    result_df = result_df.orderBy('PHYSICALID')
    #the .agg changes the order and names of the columns, select based on new name and chronologically
    result_df = result_df.select('PHYSICALID', 'sum(2015)', 'sum(2016)', 'sum(2017)', 'sum(2018)', 'sum(2019)')
    result_df = result_df.na.fill(0)
    
    #statsmodels functionality
    #slope function with 5 inputs (s, l, o, p, e) to represent the 5 cols (each phys id, yearly counts)
    # X vals are hard coded as the years
    def slope(s, l, o, p, e):
        X = ([2015, 2016, 2017, 2018, 2019])
        X = sm.add_constant(X)
        y = ([s, l, o, p, e])
        model = sm.OLS(y,X)
        results = model.fit()
        return((results.params[1]))
    
    #create a new column that is the OLS coef for each phys id, calling the slope function created, this works as slode is not a udf
    result_df = result_df.withColumn('OLS', slope(result_df['sum(2015)'], result_df['sum(2016)'], result_df['sum(2017)'], 
                                               result_df['sum(2018)'], result_df['sum(2019)']))
    #test comment   
    #order by physID and write output to csv
    result_df = result_df.orderBy('PHYSICALID')
    result_df.write.csv(sys.argv[1])
    
    starttime = datetime.now()
    elapsed = datetime.now() - starttime
    print("Total Time Elapsed: {}".format(elapsed))