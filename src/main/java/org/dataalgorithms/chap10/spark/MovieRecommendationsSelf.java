package org.dataalgorithms.chap10.spark;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;


import java.util.List;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedMap;
import java.util.Collections;
import java.util.Comparator;

import org.apache.spark.SparkConf;

class SparkTupleComparator1 
	implements Comparator<Tuple2<Integer, Double>>, Serializable {
	 
	public static final SparkTupleComparator1 INSTANCE = new SparkTupleComparator1();
	
	private SparkTupleComparator1() {
	}
	
	@Override
	public int compare(Tuple2<Integer, Double> t1, Tuple2<Integer, Double> t2){
	   return -(t1._2.compareTo(t2._2));
	}
}
public class MovieRecommendationsSelf {
	  public static void main(String[] args) throws Exception {

		if (args.length < 4) {
		   System.err.println("Input: input_files, rank, numIterations, output_model");
		   System.exit(1);
		}
		    
		SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		jsc.setCheckpointDir(args[4]);
		
		// Load and parse the data
		String path = args[0];
		JavaRDD<String> data = jsc.textFile(path);
		JavaRDD<Rating> ratings = data.map(
		  new Function<String, Rating>() {
		    public Rating call(String s) {
		      String[] sarray = s.split("\t");
		      return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
		        Double.parseDouble(sarray[2]));
		    }
		  }
		);
		
		/*
		//BEGIN/////////////////////////////////////////////////
		JavaRDD<Integer> user = data.map(
			new Function<String, Integer>() {
			    public Integer call(String s) {
				    String[] sarray = s.split("\t");
				    return Integer.parseInt(sarray[0]);
			    }
			}
		).distinct();
		
		JavaRDD<Integer> item = data.map(
			new Function<String, Integer>() {
				public Integer call(String s) {
					String[] sarray = s.split("\t");
					return Integer.parseInt(sarray[1]);
				}
			}
		).distinct();
		
		List<Integer> item_broadcast = jsc.broadcast(user.take(100)).value();
		
		JavaRDD<Tuple2<Object, Object>> userMusic = item.flatMap(
			new FlatMapFunction<Integer, Tuple2<Object, Object>>() {
				public Iterator<Tuple2<Object, Object>> call(Integer i) {
					List<Tuple2<Object, Object>> list = new ArrayList<Tuple2<Object, Object>>();
					for (Integer j : item_broadcast) {
						Tuple2 t = new Tuple2<Integer,Integer>(j,i);
				        list.add(t);
					}
					return list.iterator();
				}
			}
		);
		
		//END/////////////////////////////////////////////////
		*/

		// Build the recommendation model using ALS
		int rank = Integer.parseInt(args[1]);
		int numIterations = Integer.parseInt(args[2]);
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);
		
		// Evaluate the model on rating data
		JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
		  new Function<Rating, Tuple2<Object, Object>>() {
		    public Tuple2<Object, Object> call(Rating r) {
		      return new Tuple2<Object, Object>(r.user(), r.product());
		    }
		  }
		);
		JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
		  model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
		    new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
		      public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
		        return new Tuple2<Tuple2<Integer, Integer>, Double>(
		          new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
		      }
		    }
		  ));
		JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
		  JavaPairRDD.fromJavaRDD(ratings.map(
		    new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
		      public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
		        return new Tuple2<Tuple2<Integer, Integer>, Double>(
		          new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
		      }
		    }
		  )).join(predictions).values();
		double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
		  new Function<Tuple2<Double, Double>, Object>() {
		    public Object call(Tuple2<Double, Double> pair) {
		      Double err = pair._1() - pair._2();
		      return err * err;
		    }
		  }
		).rdd()).mean();
		System.out.println("Mean Squared Error = " + MSE);
		
		/*
		//BEGIN/////////////////////////////////////////////////
		JavaPairRDD<Integer, Tuple2<Integer, Double>> predictionsSample = JavaPairRDD.fromJavaRDD(
			model.predict(JavaRDD.toRDD(userMusic)).toJavaRDD().map(
				new Function<Rating, Tuple2<Integer, Tuple2<Integer, Double>>>() {
				   public Tuple2<Integer, Tuple2<Integer, Double>> call(Rating r){
				       return new Tuple2<Integer, Tuple2<Integer, Double>>(r.user(),
				       new Tuple2<Integer, Double>(r.product(), r.rating()));
				   }
				}
	    ));
		
		JavaPairRDD<Integer, Iterable<Tuple2<Integer, Double>>> predictionsGroup = predictionsSample.groupByKey();
		
		JavaPairRDD<Integer, Iterable<Tuple2<Integer, Double>>> predictionsSort = predictionsGroup.mapValues(
			new Function<Iterable<Tuple2<Integer, Double>>, Iterable<Tuple2<Integer, Double>>>() {
				public Iterable<Tuple2<Integer, Double>> call(Iterable<Tuple2<Integer, Double>> list) {
					List<Tuple2<Integer, Double>> list_new = new ArrayList<Tuple2<Integer, Double>>(iterableToList(list));
					Collections.sort(list_new, SparkTupleComparator1.INSTANCE);
					return list_new;
				}
			}
		);
		
		JavaPairRDD<Integer,Tuple2<Integer,Double>> usersSample = data.flatMapToPair(
				new PairFlatMapFunction<String,Integer,Tuple2<Integer,Double>>() {
					public Iterator<Tuple2<Integer,Tuple2<Integer,Double>>> call(String s) {
					    String[] sarray = s.split("\t");
					    Integer u = Integer.parseInt(sarray[0]);
					    Integer m = Integer.parseInt(sarray[1]);
					    Double c = Double.parseDouble(sarray[2]);
					    ArrayList<Tuple2<Integer,Tuple2<Integer,Double>>> list = new ArrayList<Tuple2<Integer,Tuple2<Integer,Double>>>();
					    if (item_broadcast.contains(u)) {
					    	Tuple2<Integer,Tuple2<Integer,Double>> t = new Tuple2<Integer,Tuple2<Integer,Double>>(
					    		u,new Tuple2<Integer,Double>(m,c));
					    	list.add(t);
					    }
					    return list.iterator();
				    }
				}
		);
		
		JavaPairRDD<Integer,Iterable<Tuple2<Integer,Double>>> usersGroup = usersSample.groupByKey();
		
		JavaPairRDD<Integer,Iterable<Tuple2<Integer,Double>>> usersSort = usersGroup.mapValues(
			new Function<Iterable<Tuple2<Integer,Double>>, Iterable<Tuple2<Integer,Double>>>() {
				public Iterable<Tuple2<Integer,Double>> call(Iterable<Tuple2<Integer,Double>> list) {
					List<Tuple2<Integer,Double>> list_new= new ArrayList<Tuple2<Integer,Double>>(iterableToList(list));
					Collections.sort(list_new, SparkTupleComparator1.INSTANCE);
					return list_new;
				}
			}
		);
		//userMusic.saveAsTextFile(args[3]+"/usermusic");
		//userProducts.saveAsTextFile(args[3]+"/userproducts");
		predictionsSort.saveAsTextFile(args[3]+"/predictionssort");
		usersSort.saveAsTextFile(args[3]+"/usersort");
		//END///////////////////////////////////////////////////
		*/
		
		// Save and load model
		model.save(jsc.sc(), args[3]+"/model");
		MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(), args[3]+"/model");

	  }
	  
	  static List<Tuple2<Integer,Double>> iterableToList(Iterable<Tuple2<Integer,Double>> iterable) {
		    List<Tuple2<Integer,Double>> list = new ArrayList<Tuple2<Integer,Double>>();
		    for (Tuple2<Integer,Double> item : iterable) {
		       list.add(item);
		    }
		    return list;
	  }
}