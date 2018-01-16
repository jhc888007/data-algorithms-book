package org.dataalgorithms.chap05.spark;

// STEP-0: import required classes and interfaces
import org.dataalgorithms.util.SparkUtil;

import scala.Tuple2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.SortedMap;
import java.util.Iterator;
import java.util.Collections;
import java.util.ArrayList;


/**
 * Assumption: for all input (K, V), K's are non-unique.
 * This class implements Top-N design pattern for N > 0.
 * The main assumption is that for all input (K, V)'s, K's
 * are non-unique. It means that you will find entries like
 * (A, 2), ..., (A, 5),...
 * 
 * This is a general top-N algorithm which will work unique
 * and non-unique keys.
 *
 * This class may be used to find bottom-N as well (by 
 * just keeping N-smallest elements in the set.
 * 
 *  Top-10 Design Pattern: “Top Ten” Structure 
 * 
 *  1. map(input) => (K, V)
 *         
 *  2. reduce(K, List<V1, V2, ..., Vn>) => (K, V), 
 *                where V = V1+V2+...+Vn
 *     now all K's are unique
 * 
 *  3. partition (K,V)'s into P partitions
 *
 *  4. Find top-N for each partition (we call this a local Top-N)
 * 
 *  5. Find Top-N from all local Top-N's
 *
 *
 * @author Mahmoud Parsian
 *
 */
public class RelativeFrequencySelfBack {

   public static void main(String[] args) throws Exception {
      if (args.length < 2) {
         System.err.println("Usage: Top10 <input-path> <output-path>");
         System.exit(1);
      }
      System.out.println("args[0]: <input-path>="+args[0]);
      System.out.println("args[1]: <output-path>="+args[1]);

      // STEP-2: create a Java Spark Context object
      JavaSparkContext ctx = SparkUtil.createJavaSparkContext();

      JavaRDD<String> lines = ctx.textFile(args[0], 1);
      lines.saveAsTextFile(args[1]+"/0");
     
      JavaRDD<String> rdd = lines.coalesce(3,true);
	  rdd.saveAsTextFile(args[1]+"/1");
       
      JavaPairRDD<String,Tuple2<String,Integer>> pairs = rdd.flatMapToPair(
         new PairFlatMapFunction<String,String,Tuple2<String,Integer>>() {
		 private static final long serialVersionUID = 1L;

		@Override
         public Iterator<Tuple2<String,Tuple2<String,Integer>>> call(String s) {
         	ArrayList<Tuple2<String,Tuple2<String,Integer>>> list = 
				new ArrayList<Tuple2<String,Tuple2<String,Integer>>>();
            String[] tokens = s.split(" ");
            int l = tokens.length;
			for (int i = 0; i < l; i++) {
				for (int j = 0; j < l; j++) {
					if (i != j) {
						list.add(new Tuple2<String,Tuple2<String,Integer>>(tokens[i],
							new Tuple2<String,Integer>(tokens[j],1)));
					}
				}
			}
            return list.iterator();
         }
      });
      pairs.saveAsTextFile(args[1]+"/2");

	  JavaPairRDD<String, Integer> appears = pairs.mapToPair(
	  	new PairFunction<Tuple2<String,Tuple2<String,Integer>>, String, Integer>() {
		private static final long serialVersionUID = 1L;

		@Override
	  	public Tuple2<String, Integer> call(Tuple2<String,Tuple2<String,Integer>> t) {
	  	   return new Tuple2<String, Integer>(t._1, t._2._2);
	  	}
	  });
	  appears.saveAsTextFile(args[1]+"/3");

	  JavaPairRDD<String, Integer> sums = appears.reduceByKey(
	  	new Function2<Integer,Integer,Integer>() {
		private static final long serialVersionUID = 1L;

		@Override
		public Integer call(Integer a, Integer b) {
			return a + b;
	  	}
	  });
	  sums.saveAsTextFile(args[1]+"/4");

	  JavaPairRDD<String,Iterable<Tuple2<String,Integer>>> total_pairs = pairs.groupByKey();
	  total_pairs.saveAsTextFile(args[1]+"/5");

      JavaPairRDD<Tuple2<String,String>,Integer> scores = total_pairs.flatMapToPair(
	  	new PairFlatMapFunction<Tuple2<String,Iterable<Tuple2<String,Integer>>>,
	  	Tuple2<String,String>,Integer>() {
		private static final long serialVersionUID = 1L;

		@Override
		public Iterator<Tuple2<Tuple2<String,String>,Integer>> call(
			Tuple2<String,Iterable<Tuple2<String,Integer>>> t) {
			String key1 = t._1;
			Iterator<Tuple2<String,Integer>> iter = t._2.iterator();
			TreeMap<String,Integer> map = new TreeMap<String,Integer>();
			while (iter.hasNext()) {
				Tuple2<String,Integer> tt = iter.next();
				if (map.containsKey(tt._1)) {
					map.put(tt._1, map.get(tt._1)+tt._2);
				} else {
					map.put(tt._1, tt._2);
				}	
			}
			ArrayList<Tuple2<Tuple2<String,String>,Integer>> list =
				new ArrayList<Tuple2<Tuple2<String,String>,Integer>>();
			for (Map.Entry<String,Integer> entry: map.entrySet()) {
				String k = entry.getKey();
				Integer v = entry.getValue();
				if (v > 0) {
					list.add(new Tuple2<Tuple2<String,String>,Integer>(
						new Tuple2<String,String>(key1, k), v));
				}
			}
			return list.iterator();
	  	}
      });
	  scores.saveAsTextFile(args[1]+"/6");

      /*
      // STEP-7: reduce frequent K's
      JavaPairRDD<String, Integer> uniqueKeys = kv.reduceByKey(new Function2<Integer, Integer, Integer>() {
         @Override
         public Integer call(Integer i1, Integer i2) {
            return i1 + i2;
         }
      });
      //uniqueKeys.saveAsTextFile("/output/3");
    
      // STEP-8: create a local top-N
      JavaRDD<SortedMap<Integer, String>> partitions = uniqueKeys.mapPartitions(
          new FlatMapFunction<Iterator<Tuple2<String,Integer>>, SortedMap<Integer, String>>() {
          @Override
          public Iterator<SortedMap<Integer, String>> call(Iterator<Tuple2<String,Integer>> iter) {
             final int N = topN.value();
             SortedMap<Integer, String> localTopN = new TreeMap<Integer, String>();
             while (iter.hasNext()) {
                Tuple2<String,Integer> tuple = iter.next();
                localTopN.put(tuple._2, tuple._1);
                // keep only top N 
                if (localTopN.size() > N) {
                   localTopN.remove(localTopN.firstKey());
                } 
             }
             return Collections.singletonList(localTopN).iterator();
          }
      });
      //partitions.saveAsTextFile("/output/4");

      // STEP-9: find a final top-N
      SortedMap<Integer, String> finalTopN = new TreeMap<Integer, String>();
      List<SortedMap<Integer, String>> allTopN = partitions.collect();
      for (SortedMap<Integer, String> localTopN : allTopN) {
         for (Map.Entry<Integer, String> entry : localTopN.entrySet()) {
             // count = entry.getKey()
             // url = entry.getValue()
             finalTopN.put(entry.getKey(), entry.getValue());
             // keep only top N 
             if (finalTopN.size() > N) {
                finalTopN.remove(finalTopN.firstKey());
             }
         }
      }
    
      // STEP-10: emit final top-N
      for (Map.Entry<Integer, String> entry : finalTopN.entrySet()) {
         System.out.println(entry.getKey() + "--" + entry.getValue());
      }
      */

      System.exit(0);
   }
}

