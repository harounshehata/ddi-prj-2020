
#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
from pyspark import SparkContext, SparkConf
from operator import eq

#datasets path on shared group directory on Ukko2. Uncomment the one which you would like to work on.
dataset = "/wrk/group/grp-ddi-2021/datasets/data-2-sample.txt"
# dataset = "/wrk/group/grp-ddi-2021/datasets/data-2.txt"

conf = (SparkConf()
        .setAppName("sheharou")
        .setMaster("spark://ukko2-10.local.cs.helsinki.fi:7077")
        .set("spark.cores.max", "10")
        .set("spark.rdd.compress", "true")
        .set("spark.broadcast.compress", "true")
        .set("spark.executor.memory", "16g"))
sc = SparkContext(conf=conf)
#sc.setLogLevel("WARN")

#data = sc.textFile(dataset)
#data = data.map(lambda v: list(map(float, v.split())))

sample_data = [[1,2,3], [4,5,6], [7,8,9]]

data = sc.parallelize(sample_data)

def add_indices(rdd):
        s1 = rdd.zipWithIndex() # row index
        s2 = s1.flatMap(lambda x: [(x[1],j,e) for (j,e) in enumerate(x[0])]) # column index
        print(f"Add indices: {s2.collect()}")
        return s2

def add_indices_2(rdd):
        s1 = rdd.zipWithIndex() # row index
        s2 = s1.flatMap(lambda x: [(x[1],(j,e)) for (j,e) in enumerate(x[0])]) # column index
        print(f"Add indices: {s2.collect()}")
        s3 = s2.groupByKey().map(lambda x : (x[0], list(x[1]))).sortByKey()
        print(f"GroupRow {s3.collect()}") # [(2, [(0, 7), (1, 8), (2, 9)]), (0, [(0, 1), (1, 2), (2, 3)]), (1, [(0, 4), (1, 5), (2, 6)])]
        return s3
# transpose matrix with indices
def transpose(rdd):
        s1 = rdd.map(lambda x: ((x[1], (x[0],x[2]))))
        s2 = s1.groupByKey().sortByKey().map(lambda x: (x[0], sorted(list(x[1]), key=lambda x: x[0])))
        return s2

def remove_indices(rdd):
        s1 = rdd.map(lambda x: x[1])
        s2 = s1.map(lambda x: list(map(lambda y: y[1], x)))
        return s2

def add_single_indices(rdd):
        s1 = rdd.zipWithIndex()
        s2 = s1.map(lambda x: (x[1], x[0]))
        return s2

def matrix_multiplication(rdd1_row_indices, rdd2_col_indices):
        print(rdd1_row_indices.collect())
        print(rdd2_col_indices.collect())
        s1 = rdd1_row_indices.cartesian(rdd2_col_indices)
        print(f"Cartesian {s1.collect()}")
        s2 = s1.map(lambda x: ((x[0][0], x[1][0]), dot_product(x[0][1], x[1][1])))
        print(f"DotProd {s2.collect()}")
        s3 = s2.sortByKey()
        print(f"Sort {s3.collect()}")
        #s4 = s3.map(lambda x: (x[0][0], (x[0][1], x[1])))
        #print(f"RowKey {s4.collect()}")
        #s5 = s4.groupByKey()
        #print(f"GroupRow {s5.collect()}")
        #s6 = s5.map(lambda x: (x[0], list(map(lambda y: y[1], list(x[1])))))
        #print(f"Map {s6.collect()}")
        return s3


def matrix_multiplication_2(rdd1_row_indices, rdd2_col_indices):
        s1 = rdd1_row_indices.cartesian(rdd2_col_indices)
        #print(f"Cartesian {s1.collect()}")
        s2 = s1.map(lambda x: ((x[0][0], x[1][0]), dot_product_with_indices(x[0][1], x[1][1])))
        s3 = s2.sortByKey()
        #print(f"Sort {s3.collect()}") # [((0, 0), 14), ((0, 1), 32), ((0, 2), 50), ((1, 0), 32), ((1, 1), 77), ((1, 2), 122), ((2, 0), 50), ((2, 1), 122), ((2, 2), 194)]
        s4 = s3.map(lambda x: (x[0][0], (x[0][1], x[1])))
        #print(f"RowKey {s4.collect()}")
        s5 = s4.groupByKey().map(lambda x : (x[0], list(x[1]))).sortByKey()
        print(f"Mult: {s5.collect()}")
        return s5

def dot_product(l1, l2):
        return sum(x*y for x,y in zip(l1,l2))

def dot_product_with_indices(l1,l2):
        return sum(x[1]*y[1] for x,y in zip(l1,l2))

data_indices = add_indices(data)
data_transposed_indices = transpose(data_indices)
data_transposed = remove_indices(data_transposed_indices)

print("mutliplication one:\n")
#A_times_A_t_res = matrix_multiplication(add_single_indices(data), add_single_indices(data))
A_times_A_t_res = matrix_multiplication_2(add_indices_2(data), add_indices_2(data))
print("mutliplication two:\n")
A_times_A_t_times_A_res = matrix_multiplication_2(A_times_A_t_res, add_indices_2(data_transposed))




