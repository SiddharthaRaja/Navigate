# Leaderboard (Hyderabad, 2026-11-16)

Trial run on a 60,000-building sample with local baselines only. Replace by re-running notebook 09 after the full build.

Full test sets:

| split     | model            |    n |   coverage |   acc@100 |   acc@500 |   acc@1000 |      p50 |      p90 |   misroute |
|:----------|:-----------------|-----:|-----------:|----------:|----------:|-----------:|---------:|---------:|-----------:|
| test_cold | city_centroid    | 2946 |      1     |     0     |     0     |      0     |  9363.4  | 16776    |      1     |
| test_cold | gazetteer_rules  | 2946 |      0.958 |     0.121 |     0.528 |      0.643 |   343.7  |  8735.8  |      0.307 |
| test_cold | oracle_reference | 2946 |      0.993 |     0.275 |     0.87  |      0.964 |   171.35 |   593.12 |    nan     |
| test_cold | pincode_centroid | 2946 |      1     |     0     |     0.017 |      0.091 |  3312.7  | 13614.5  |      0.522 |
| test_cold | tfidf_knn        | 2946 |      0.997 |     0.007 |     0.144 |      0.292 |  4070.35 | 18291.3  |      0.623 |
| test_real | city_centroid    | 1379 |      1     |     0     |     0     |      0.001 | 12885.4  | 23885.5  |    nan     |
| test_real | gazetteer_rules  | 1379 |      0.787 |     0.286 |     0.468 |      0.563 |   243.7  |  2528.22 |    nan     |
| test_real | pincode_centroid | 1379 |      1     |     0.006 |     0.066 |      0.216 |  2112.4  | 23867.3  |    nan     |
| test_real | tfidf_knn        | 1379 |      0.98  |     0.026 |     0.335 |      0.417 |  2209.1  | 24597.4  |    nan     |
| test_rel  | city_centroid    | 1269 |      1     |     0     |     0.001 |      0.003 |  8833.4  | 15472.1  |      0.999 |
| test_rel  | gazetteer_rules  | 1269 |      0.975 |     0.372 |     0.697 |      0.753 |   140.6  |  9861.5  |      0.252 |
| test_rel  | oracle_reference | 1269 |      1     |     0.676 |     1     |      1     |    67.9  |   186.82 |    nan     |
| test_rel  | pincode_centroid | 1269 |      1     |     0     |     0.025 |      0.117 |  3743.3  | 13251.5  |      0.572 |
| test_rel  | tfidf_knn        | 1269 |      0.995 |     0.029 |     0.23  |      0.336 |  4228    | 17745.9  |      0.632 |
| test_warm | city_centroid    | 2939 |      1     |     0     |     0.001 |      0.007 | 11074    | 21099.2  |      1     |
| test_warm | gazetteer_rules  | 2939 |      0.958 |     0.137 |     0.579 |      0.689 |   290.8  |  5652.04 |      0.277 |
| test_warm | oracle_reference | 2939 |      0.956 |     0.271 |     0.863 |      0.936 |   169.25 |   496.14 |    nan     |
| test_warm | pincode_centroid | 2939 |      1     |     0.005 |     0.076 |      0.21  |  3407.9  | 17382.1  |      0.529 |
| test_warm | tfidf_knn        | 2939 |      1     |     0.184 |     0.757 |      0.851 |   239.6  |  2911.36 |      0.19  |

API subset (same rows for every system):

| split     | model            |   n |   coverage |   acc@100 |   acc@500 |   acc@1000 |      p50 |      p90 |   misroute |
|:----------|:-----------------|----:|-----------:|----------:|----------:|-----------:|---------:|---------:|-----------:|
| test_cold | city_centroid    | 900 |      1     |     0     |     0     |      0     |  9210.15 | 16506.7  |      1     |
| test_cold | gazetteer_rules  | 900 |      0.956 |     0.161 |     0.561 |      0.652 |   288.35 | 10668.9  |      0.31  |
| test_cold | oracle_reference | 900 |      0.996 |     0.37  |     0.894 |      0.977 |   156.5  |   502.7  |    nan     |
| test_cold | pincode_centroid | 900 |      1     |     0     |     0.021 |      0.071 |  4640.05 | 14150.5  |      0.614 |
| test_cold | tfidf_knn        | 900 |      1     |     0.006 |     0.119 |      0.263 |  5767.65 | 18862.3  |      0.672 |
| test_real | city_centroid    | 300 |      1     |     0     |     0     |      0.003 | 14956.9  | 23883.4  |    nan     |
| test_real | gazetteer_rules  | 300 |      0.77  |     0.283 |     0.483 |      0.557 |   190.8  |  2538.2  |    nan     |
| test_real | oracle_reference | 300 |      0     |     0     |     0     |      0     |   nan    |   nan    |    nan     |
| test_real | pincode_centroid | 300 |      1     |     0.003 |     0.073 |      0.203 |  2914.35 | 23868.1  |    nan     |
| test_real | tfidf_knn        | 300 |      0.967 |     0.023 |     0.327 |      0.417 |  3317.85 | 22515.6  |    nan     |
| test_rel  | city_centroid    | 450 |      1     |     0     |     0.002 |      0.007 |  9505.8  | 16937.8  |      0.998 |
| test_rel  | gazetteer_rules  | 450 |      0.978 |     0.416 |     0.689 |      0.753 |   131.75 | 10097.8  |      0.266 |
| test_rel  | oracle_reference | 450 |      1     |     0.727 |     1     |      1     |    53.8  |   180.66 |    nan     |
| test_rel  | pincode_centroid | 450 |      1     |     0     |     0.013 |      0.111 |  3825.35 | 13684.7  |      0.576 |
| test_rel  | tfidf_knn        | 450 |      0.996 |     0.04  |     0.231 |      0.34  |  3568.15 | 17696.7  |      0.616 |
| test_warm | city_centroid    | 900 |      1     |     0     |     0.001 |      0.009 | 11070.4  | 21310    |      0.999 |
| test_warm | gazetteer_rules  | 900 |      0.943 |     0.199 |     0.62  |      0.722 |   251.7  |  6560.64 |      0.254 |
| test_warm | oracle_reference | 900 |      0.976 |     0.379 |     0.899 |      0.959 |   140.95 |   440.56 |    nan     |
| test_warm | pincode_centroid | 900 |      1     |     0.004 |     0.062 |      0.188 |  4739.5  | 18452.1  |      0.604 |
| test_warm | tfidf_knn        | 900 |      1     |     0.168 |     0.748 |      0.833 |   251.1  |  5964.46 |      0.221 |
