# Leaderboard (Hyderabad, 2026-11-18)

Baselines and open geocoders only. Trained models and paid geocoders are added as they are run.

Full test sets:

| split     | model            |    n |   coverage |   acc@100 |   acc@500 |   acc@1000 |      p50 |      p90 |   misroute |
|:----------|:-----------------|-----:|-----------:|----------:|----------:|-----------:|---------:|---------:|-----------:|
| test_cold | city_centroid    | 2936 |      1     |     0     |     0     |      0     |  9896.6  | 16769.6  |      1     |
| test_cold | gazetteer_rules  | 2936 |      0.958 |     0.166 |     0.557 |      0.656 |   291.6  |  7778.82 |      0.308 |
| test_cold | nominatim        |  900 |      0.104 |     0.018 |     0.063 |      0.074 |   285.9  |  8240.75 |      0.287 |
| test_cold | oracle_reference | 2936 |      0.989 |     0.353 |     0.913 |      0.981 |   146.8  |   398.02 |    nan     |
| test_cold | photon           |  900 |      0.212 |     0.028 |     0.116 |      0.141 |   399.6  | 13798.7  |      0.361 |
| test_cold | pincode_centroid | 2936 |      1     |     0.001 |     0.022 |      0.101 |  4392.1  | 13594.1  |      0.587 |
| test_cold | tfidf_knn        | 2936 |      0.999 |     0.006 |     0.147 |      0.299 |  3707.5  | 18934    |      0.624 |
| test_real | city_centroid    | 2879 |      1     |     0     |     0.001 |      0.006 | 10299.2  | 22832.5  |    nan     |
| test_real | gazetteer_rules  | 2879 |      0.97  |     0.18  |     0.372 |      0.522 |   842.5  | 14234.8  |    nan     |
| test_real | nominatim        |  300 |      0.383 |     0.313 |     0.353 |      0.357 |     0    |   401.64 |    nan     |
| test_real | photon           |  300 |      0.523 |     0.39  |     0.427 |      0.44  |     0    |  3468.16 |    nan     |
| test_real | pincode_centroid | 2879 |      1     |     0.009 |     0.123 |      0.33  |  1358.8  | 22801    |    nan     |
| test_real | tfidf_knn        | 2879 |      0.99  |     0.04  |     0.305 |      0.427 |  1498.7  | 19533.1  |    nan     |
| test_rel  | city_centroid    | 1750 |      1     |     0     |     0.001 |      0.003 |  9832.5  | 20337.9  |      0.999 |
| test_rel  | gazetteer_rules  | 1750 |      0.957 |     0.379 |     0.65  |      0.707 |   150.3  |  7368.11 |      0.25  |
| test_rel  | nominatim        |  450 |      0.009 |     0.002 |     0.007 |      0.007 |   151.1  | 13042.3  |      0.75  |
| test_rel  | oracle_reference | 1750 |      1     |     0.733 |     0.999 |      1     |    50.5  |   167.45 |    nan     |
| test_rel  | photon           |  450 |      0.069 |     0.02  |     0.036 |      0.044 |   300.2  | 13072.4  |      0.355 |
| test_rel  | pincode_centroid | 1750 |      1     |     0.001 |     0.043 |      0.152 |  4233.8  | 13921.9  |      0.588 |
| test_rel  | tfidf_knn        | 1750 |      0.997 |     0.077 |     0.311 |      0.401 |  2099.85 | 18644.8  |      0.552 |
| test_warm | city_centroid    | 2938 |      1     |     0     |     0.003 |      0.007 | 10818    | 21172.8  |      0.999 |
| test_warm | gazetteer_rules  | 2938 |      0.96  |     0.174 |     0.582 |      0.673 |   278.9  |  9172.87 |      0.288 |
| test_warm | nominatim        |  900 |      0.102 |     0.014 |     0.057 |      0.076 |   413.35 |  9624.63 |      0.326 |
| test_warm | oracle_reference | 2938 |      0.98  |     0.34  |     0.917 |      0.966 |   144.3  |   379.5  |    nan     |
| test_warm | photon           |  900 |      0.202 |     0.023 |     0.093 |      0.126 |   620.4  | 13820    |      0.407 |
| test_warm | pincode_centroid | 2938 |      1     |     0.005 |     0.078 |      0.21  |  3285.15 | 17258    |      0.548 |
| test_warm | tfidf_knn        | 2938 |      1     |     0.209 |     0.798 |      0.866 |   222.8  |  3353.26 |      0.167 |

API subset (same rows for every system):

| split     | model            |   n |   coverage |   acc@100 |   acc@500 |   acc@1000 |      p50 |      p90 |   misroute |
|:----------|:-----------------|----:|-----------:|----------:|----------:|-----------:|---------:|---------:|-----------:|
| test_cold | city_centroid    | 900 |      1     |     0     |     0     |      0     |  9753.5  | 14806.4  |      1     |
| test_cold | gazetteer_rules  | 900 |      0.958 |     0.209 |     0.583 |      0.669 |   270.15 |  9517.32 |      0.309 |
| test_cold | nominatim        | 900 |      0.104 |     0.018 |     0.063 |      0.074 |   285.9  |  8240.75 |      0.287 |
| test_cold | oracle_reference | 900 |      0.988 |     0.448 |     0.93  |      0.983 |   115.9  |   324.42 |    nan     |
| test_cold | photon           | 900 |      0.212 |     0.028 |     0.116 |      0.141 |   399.6  | 13798.7  |      0.361 |
| test_cold | pincode_centroid | 900 |      1     |     0     |     0.017 |      0.088 |  4822.35 | 13648.6  |      0.641 |
| test_cold | tfidf_knn        | 900 |      0.999 |     0.007 |     0.134 |      0.284 |  5577.3  | 19702.2  |      0.655 |
| test_real | city_centroid    | 300 |      1     |     0     |     0     |      0.01  |  9944.55 | 22778.6  |    nan     |
| test_real | gazetteer_rules  | 300 |      0.983 |     0.203 |     0.42  |      0.553 |   730.8  |  7326.6  |    nan     |
| test_real | nominatim        | 300 |      0.383 |     0.313 |     0.353 |      0.357 |     0    |   401.64 |    nan     |
| test_real | photon           | 300 |      0.523 |     0.39  |     0.427 |      0.44  |     0    |  3468.16 |    nan     |
| test_real | pincode_centroid | 300 |      1     |     0.01  |     0.123 |      0.34  |  1281.55 | 22757    |    nan     |
| test_real | tfidf_knn        | 300 |      0.983 |     0.047 |     0.327 |      0.457 |  1249.4  | 21127.6  |    nan     |
| test_rel  | city_centroid    | 450 |      1     |     0     |     0.002 |      0.007 |  9973.65 | 19789.4  |      1     |
| test_rel  | gazetteer_rules  | 450 |      0.971 |     0.371 |     0.664 |      0.724 |   147.6  |  6862.76 |      0.24  |
| test_rel  | nominatim        | 450 |      0.009 |     0.002 |     0.007 |      0.007 |   151.1  | 13042.3  |      0.75  |
| test_rel  | oracle_reference | 450 |      1     |     0.727 |     1     |      1     |    48.45 |   162.27 |    nan     |
| test_rel  | photon           | 450 |      0.069 |     0.02  |     0.036 |      0.044 |   300.2  | 13072.4  |      0.355 |
| test_rel  | pincode_centroid | 450 |      1     |     0     |     0.04  |      0.162 |  4221.7  | 13245.3  |      0.584 |
| test_rel  | tfidf_knn        | 450 |      0.998 |     0.051 |     0.258 |      0.349 |  2852.3  | 19508.5  |      0.586 |
| test_warm | city_centroid    | 900 |      1     |     0     |     0.002 |      0.006 | 10682.4  | 21496.6  |      0.998 |
| test_warm | gazetteer_rules  | 900 |      0.946 |     0.232 |     0.603 |      0.667 |   259.9  | 10632.3  |      0.298 |
| test_warm | nominatim        | 900 |      0.102 |     0.014 |     0.057 |      0.076 |   413.35 |  9624.63 |      0.326 |
| test_warm | oracle_reference | 900 |      0.984 |     0.419 |     0.926 |      0.971 |   123.9  |   363.3  |    nan     |
| test_warm | photon           | 900 |      0.202 |     0.023 |     0.093 |      0.126 |   620.4  | 13820    |      0.407 |
| test_warm | pincode_centroid | 900 |      1     |     0.007 |     0.07  |      0.163 |  4739.95 | 17511.2  |      0.63  |
| test_warm | tfidf_knn        | 900 |      1     |     0.199 |     0.803 |      0.867 |   226.2  |  2578.24 |      0.174 |
