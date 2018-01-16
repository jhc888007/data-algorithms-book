package org.dataalgorithms.chap10.spark;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.Partitioner;

import java.util.List;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Collections;
import java.util.Comparator;

import org.apache.spark.SparkConf;

class SparkMovieTupleComparator1 
	implements Comparator<Tuple2<Integer, Tuple2<Integer, Double>>>, Serializable {
	 
	public static final SparkMovieTupleComparator1 INSTANCE = new SparkMovieTupleComparator1();
	
	private SparkMovieTupleComparator1() {
	}
	
	@Override
	public int compare(Tuple2<Integer, Tuple2<Integer, Double>> t1, Tuple2<Integer, Tuple2<Integer, Double>> t2){
	   return -(t1._2._2.compareTo(t2._2._2));
	}
}

public class MovieRecommendationsSelfPredict1 {
	  public static void main(String[] args) throws Exception {

		if (args.length < 4) {
		   System.err.println("Input: input_files, rank, numIterations, output_model");
		   System.exit(1);
		}
		    
		SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		
		// Load and parse the data
		String path = args[0];
		final int Part = Integer.parseInt(args[3]);
		JavaRDD<String> data = jsc.textFile(path, Part);
		
		   //BEGIN/////////////////////////////////////////////////
		/*
		   JavaPairRDD<Integer, Tuple2<Integer, Double>> userScore = data.mapToPair(
				new PairFunction<String, Integer, Tuple2<Integer, Double>>() {
					public Tuple2<Integer, Tuple2<Integer, Double>> call(String s) {
						String[] sarray = s.split("\t");
						return new Tuple2<Integer, Tuple2<Integer, Double>>
							(Integer.parseInt(sarray[0]), 
							new Tuple2<Integer, Double>(Integer.parseInt(sarray[1]),
							Double.parseDouble(sarray[2])));
					}
				}
			);
			System.out.printf("userScore %d\n", userScore.count());
		
			JavaPairRDD<Integer, Iterable<Tuple2<Integer, Double>>> userScoreGroup = userScore.groupByKey();
			System.out.printf("userScoreGroup %d\n", userScoreGroup.count());
			
			JavaRDD<Tuple2<Integer, SortedMap<Double, Integer>>> userScoreSort = userScoreGroup.map(
				new Function<Tuple2<Integer, Iterable<Tuple2<Integer, Double>>>,
					Tuple2<Integer, SortedMap<Double, Integer>>>() {
					@Override
					public Tuple2<Integer, SortedMap<Double, Integer>> call(
						Tuple2<Integer, Iterable<Tuple2<Integer, Double>>> tu) {
						Iterator<Tuple2<Integer, Double>> iter = tu._2.iterator();
						SortedMap<Double, Integer> map = new TreeMap<Double, Integer>();
			            while (iter.hasNext()) {
			                 Tuple2<Integer, Double> t = iter.next();
			                 map.put(t._2, t._1);
			            }
						return new Tuple2<Integer, SortedMap<Double, Integer>>(tu._1, map);
					}
				}
			);
			System.out.printf("userScoreSort %d\n", userScoreSort.count());

			userScoreSort.saveAsTextFile(args[5]+"/userscoresort");
			System.out.println("userScoreSort saved");
			*/
			//END/////////////////////////////////////////////////

		//BEGIN/////////////////////////////////////////////////
		JavaRDD<Tuple2<Integer, Integer>> userMusicHistory = data.map(
			new Function<String, Tuple2<Integer, Integer>>() {
				public Tuple2<Integer, Integer> call(String s) {
					String[] sarray = s.split("\t");
					return new Tuple2<Integer, Integer>
						(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]));
				}
			}
		);
		System.out.printf("userMusicHistory %d\n", userMusicHistory.count());
		
		JavaRDD<Integer> userAll = userMusicHistory.map(
			new Function<Tuple2<Integer, Integer>, Integer>() {
			    public Integer call(Tuple2<Integer, Integer> t) {
				    return t._1;
			    }
			}
		).distinct();
		System.out.printf("userAll %d\n", userAll.count());
		
		JavaRDD<Integer> itemAll = userMusicHistory.map(
			new Function<Tuple2<Integer, Integer>, Integer>() {
				public Integer call(Tuple2<Integer, Integer> t) {
					return t._2;
				}
			}
		).distinct();
		System.out.printf("itemAll %d\n", itemAll.count());
		
		final int sampleNum = Integer.parseInt(args[1]);
		List<Integer> userBroadcast = jsc.broadcast(userAll.take(sampleNum)).value();
		
		/*
		JavaRDD<Tuple2<Integer, Integer>> userMusicAll = itemAll.flatMap(
			new FlatMapFunction<Integer, Tuple2<Integer, Integer>>() {
				public Iterator<Tuple2<Integer, Integer>> call(Integer i) {
					List<Tuple2<Integer, Integer>> list = new ArrayList<Tuple2<Integer, Integer>>();
					for (Integer j : userBroadcast) {
						Tuple2 t = new Tuple2<Integer,Integer>(j,i);
				        list.add(t);
					}
					return list.iterator();
				}
			}
		);
		System.out.printf("userMusicAll %d\n", userMusicAll.count());
		
		JavaRDD<Tuple2<Integer, Integer>> userMusicSample = userMusicAll.subtract(userMusicHistory);
		System.out.printf("userMusicSample %d\n", userMusicSample.count());
		
		
		JavaRDD<Tuple2<Object, Object>> userMusicPredict = userMusicAll.map(
			new Function<Tuple2<Integer, Integer>, Tuple2<Object, Object>>() {
				public Tuple2<Object, Object> call(Tuple2<Integer, Integer> t) {
					return new Tuple2<Object, Object>(t._1, t._2);
				}
			}
		);
		System.out.printf("userMusicPredict %d\n", userMusicPredict.count());
		*/
		
		JavaRDD<Tuple2<Object, Object>> userMusicPredict = itemAll.flatMap(
				new FlatMapFunction<Integer, Tuple2<Object, Object>>() {
					public Iterator<Tuple2<Object, Object>> call(Integer i) {
						List<Tuple2<Object, Object>> list = new ArrayList<Tuple2<Object, Object>>();
						for (Integer j : userBroadcast) {
							Tuple2 t = new Tuple2<Integer,Integer>(j,i);
					        list.add(t);
						}
						return list.iterator();
					}
				}
			);
		System.out.printf("userMusicPredict %d\n", userMusicPredict.count());	
		
		MatrixFactorizationModel model = MatrixFactorizationModel.load(jsc.sc(), args[5]+"/model");
		System.out.println("model");
		
		JavaPairRDD<Integer, Tuple2<Integer, Double>> userMusicResult = JavaPairRDD.fromJavaRDD(
			model.predict(JavaRDD.toRDD(userMusicPredict)).toJavaRDD().map(
				new Function<Rating, Tuple2<Integer, Tuple2<Integer, Double>>>() {
				   public Tuple2<Integer, Tuple2<Integer, Double>> call(Rating r){
				       return new Tuple2<Integer, Tuple2<Integer, Double>>(r.user(),
				       new Tuple2<Integer, Double>(r.product(), r.rating()));
				   }
				}
	    ));
		System.out.printf("userMusicResult %d\n", userMusicResult.count());
		//userMusicResult.saveAsTextFile(args[4]+"/usermusicresult");
		//System.out.println("userMusicResult saved");
		
		final double threshod = Double.parseDouble(args[4]);
		JavaPairRDD<Integer, Tuple2<Integer, Double>> userMusicFilter = userMusicResult.filter(
			new Function<Tuple2<Integer, Tuple2<Integer, Double>>,Boolean>() {
				public Boolean call(Tuple2<Integer, Tuple2<Integer, Double>> t) {
					if (t._2._2 < 1.0) {
						return false;
					}
					return true;
				}
			}
		);
		System.out.printf("userMusicFilter %d\n", userMusicFilter.count());
		
		JavaPairRDD<Integer, Iterable<Tuple2<Integer, Double>>> userMusicGroup = userMusicFilter.groupByKey();
		System.out.printf("userMusicGroup %d\n", userMusicGroup.count());
		
		final int N = Integer.parseInt(args[2]);
		JavaRDD<Tuple2<Integer, SortedMap<Double, Integer>>> userMusicSort = userMusicGroup.map(
			new Function<Tuple2<Integer, Iterable<Tuple2<Integer, Double>>>,
				Tuple2<Integer, SortedMap<Double, Integer>>>() {
				@Override
				public Tuple2<Integer, SortedMap<Double, Integer>> call(
					Tuple2<Integer, Iterable<Tuple2<Integer, Double>>> tu) {
					Iterator<Tuple2<Integer, Double>> iter = tu._2.iterator();
					SortedMap<Double, Integer> map = new TreeMap<Double, Integer>();
		            while (iter.hasNext()) {
		                 Tuple2<Integer, Double> t = iter.next();
		                 map.put(t._2, t._1);
		                 if (map.size() > N) {
		                	 map.remove(map.firstKey());
		                 } 
		            }
					return new Tuple2<Integer, SortedMap<Double, Integer>>(tu._1, map);
				}
			}
		);
		System.out.printf("userMusicSort %d\n", userMusicSort.count());
		
		userMusicSort.saveAsTextFile(args[5]+"/usermusicsort");
		System.out.println("userMusicSort saved");
		
		//END///////////////////////////////////////////////////
	  }
	  
	  static List<Tuple2<Integer,Double>> iterableToList(Iterable<Tuple2<Integer,Double>> iterable) {
		    List<Tuple2<Integer,Double>> list = new ArrayList<Tuple2<Integer,Double>>();
		    for (Tuple2<Integer,Double> item : iterable) {
		       list.add(item);
		    }
		    return list;
	  }
}
