
#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
from pyspark import SparkContext, SparkConf
from operator import eq

#datasets path on shared group directory on Ukko2. Uncomment the one which you would like to work on.
dataset = "/wrk/group/grp-ddi-2021/datasets/data-2-sample.txt"
# dataset = "/wrk/group/grp-ddi-2021/datasets/data-2.txt"

conf = (SparkConf()
        .setAppName("pfabel")
        .setMaster("spark://ukko2-10.local.cs.helsinki.fi:7077")
        .set("spark.cores.max", "10")
        .set("spark.rdd.compress", "true")
        .set("spark.broadcast.compress", "true")
        .set("spark.executor.memory", "16g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

# data = sc.textFile(dataset)
# data = data.map(lambda v: list(map(float, v.split())))

sample_data = [[1,2,3], [4,5,6], [7,8,9]]
sample_data_col = [[1,4,7], [2,5,8], [3,6,9]]

data = sc.parallelize(sample_data)

def add_indices(rdd):
        s1 = rdd.zipWithIndex() # row index
        s2 = s1.flatMap(lambda x: [(x[1],j,e) for (j,e) in enumerate(x[0])]) # column index
        print(s2)
        return s2

# transpose matrix with indices
def transpose(rdd):
        s1 = rdd.map(lambda x: ((x[1], (x[0],x[2]))))
        s2 = s1.groupByKey()
        s3 = s2.sortByKey()
        s4 = s3.map(lambda x: (x[0], sorted(list(x[1]), key=lambda x: x[0])))
        return s4

def remove_indices(rdd):
        s1 = rdd.map(lambda x: x[1])
        s2 = s1.map(lambda x: list(map(lambda y: y[1], x)))
        return s2

data_indices = add_indices(data)
data_transposed_indices = transpose(data_indices)
data_transposed = remove_indices(data_transposed_indices)
print("Data transposed:")
print(data_transposed.collect())

def add_single_indices(rdd):
        s1 = rdd.zipWithIndex()
        s2 = s1.map(lambda x: (x[1], x[0]))
        return s2

def matrix_multiplication(rdd1_row_indices, rdd2_col_indices):
        print(rdd1_row_indices.collect())
        print(rdd2_col_indices.collect())
        s1 = rdd1_row_indices.cartesian(rdd2_col_indices)
        print(s1.collect())
        s2 = s1.map(lambda x: ((x[0][0], x[1][0]), dot_product(x[0][1], x[1][1])))
        print(s2.collect())
        s3 = s2.sortByKey()
        print(s3.collect())
        s4 = s3.map(lambda x: (x[0][0], (x[0][1], x[1])))
        print(s4.collect())
        s5 = s4.groupByKey()
        print(s5.collect())
        s6 = s5.map(lambda x: (x[0], list(map(lambda y: y[1], list(x[1])))))
        print(s6.collect())
        return s6

def dot_product(l1, l2):
        return sum(x*y for x,y in zip(l1,l2))

print("mutliplication one:\n")
A_times_A_t_res = matrix_multiplication(add_single_indices(data), add_single_indices(data))
add_indices(data)
print("mutliplication two:\n")
A_times_A_t_times_A_res = matrix_multiplication(A_times_A_t_res, add_single_indices(data_transposed))




