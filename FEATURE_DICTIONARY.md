# Feature Dictionary

This file explains the two feature files, enriched_features_train.parquet and enriched_features_ranking.parquet.

Prepared by Harsh, Data Enrichment.

Each row is one departing flight. Each flight has an ID number called MVT_ID_mvt, and every ID is different. To use this data, join it to the cleaned flight data by matching on MVT_ID_mvt.

## The time the features are based on

The test file does not have the real pushback time. The column BLOCK_TIME_UTC_mvt is empty for every test flight. So the features are based on the actual off block time instead, which is the column AOBT_3_flt. For the few flights that do not have that time, the scheduled time (SCHED_TIME_UTC_mvt) is used. Both of these times are in the training file and the test file, so the features match across the two.

Every feature uses only information known at or before that off block time. Nothing uses the takeoff time or the taxi out time.

## The columns

| Column | What it means |
|--------|----------------|
| MVT_ID_mvt | The flight's ID number. Used to match this data to the flight data. |
| pushback_hour | The hour of the day the flight left the gate, from 0 to 23. |
| is_weekend | 1 if the flight left on a Saturday or Sunday, otherwise 0. |
| planes_pushed_last_15min | How many flights left their gate at the same airport in the 15 minutes before this flight. |
| planes_pushed_last_30min | The same count for the last 30 minutes. |
| planes_pushed_last_60min | The same count for the last 60 minutes. |
| visibility_miles | How far you could see at the airport, in miles. |
| wind_kt | The wind speed at the airport, in knots. |
| temp_c | The air temperature at the airport, in degrees Celsius. |
| is_fog | 1 if there was fog, otherwise 0. |
| is_snow_ice | 1 if there was snow or freezing weather, otherwise 0. This points to deicing conditions. |
| stand_area | A rough parking zone, taken from the first letter of the stand code. |

## Things to know

1. The test file has no real pushback time, so nothing should use BLOCK_TIME_UTC_mvt at prediction time.
2. Please do not use MVT_TIME_UTC_mvt, the takeoff time, in the model. It is not known in advance, so it would make the model wrong.
3. Read stand_area together with the airport. The same letter means a different place at different airports.
4. The snow, ice and temperature columns matter because the test months include January, when deicing is common and taxi times grow.
5. A small number of flights have no weather value, about 84 in the whole training year. Treat those as empty.
6. Rain and snow amounts were left out on purpose, because the weather source does not report them reliably for airports outside the United States.

## How the counts and the weather stay fair

Each traffic count looks only at the minutes before the flight, and it does not count the flight itself. The weather is taken from the most recent reading at or before the flight's off block time, never a later one.

## How to run it again

The weather is downloaded once and saved in weather_cache.parquet, which is in the project. The code reads that saved file instead of going online, so it runs the same on any computer. The code needs pandas, pyarrow and requests.

The build needs these columns from the flight data: MVT_ID_mvt, ADEP_mvt, STAND_mvt, AOBT_3_flt, SCHED_TIME_UTC_mvt, and PHASE_mvt.
