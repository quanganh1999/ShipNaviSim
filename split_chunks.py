import glob
import polars as pl
from datetime import timedelta

# Columns: SHIP_ID;MMSI;IMO;LAT;LON;TIMESTAMP_UTC;STATUS;HEADING;COURSE;SPEED_KNOTSX10
data_path = "./raw_data"
files = glob.glob(f"{data_path}/*Positions*.csv")
files = sorted(files)

SHIP_ID_COL = "SHIP_ID"
TIME_COL = "TIMESTAMP_UTC"

def seperate_chunks(df, time_theshold, minimum_sample_count):
    df = df.lazy()
    df = df.sort([SHIP_ID_COL, TIME_COL])

    # Create a new column for the time difference
    df = df.with_columns([
        pl.col(TIME_COL).diff().dt.cast_time_unit('ms').alias('time_diff'),
        pl.col(SHIP_ID_COL).diff().alias('ship_id_diff')
    ])
    
    # Create a chunk_id column
    df = df.with_columns([
        pl.when((pl.col('time_diff') > time_theshold) | (pl.col('ship_id_diff') != 0))
        .then(1)
        .otherwise(0)
        .cum_sum()
        .alias('chunk_id')
    ])
        
    # Group by chunk_id and aggregate
    result = df.group_by('chunk_id').agg([
        pl.col(SHIP_ID_COL).first().alias(SHIP_ID_COL),
        pl.col(TIME_COL).min().alias('start_time'),
        pl.col(TIME_COL).max().alias('end_time'),
        pl.count(TIME_COL).alias('num_records'),
        pl.struct(pl.all()).alias('all_records')
    ]).sort([SHIP_ID_COL, 'start_time'])
    result = result.filter(pl.col('num_records')>=minimum_sample_count).collect()
    return result
