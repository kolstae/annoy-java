package com.spotify.annoy;

import java.io.IOException;
import java.util.Random;


// assumes there are 1M points in the index

public class SpeedTest {

   public static void testSpeed(String indexPath, int dimension, int nQueries) throws IOException {
      try (ANNIndex index = new ANNIndex(dimension, indexPath)) {
         Random r = new Random();
         // float[] itemVector= new float[dimension];

         long tStart = System.currentTimeMillis();
         for (int i = 0; i < nQueries; i++) {
            int k = Math.abs(r.nextInt() % 1000000);
            // System.out.println("querying with item " + k);
            index.getNearest(k, x -> true, 10);
         }
         long tEnd = System.currentTimeMillis();

         System.out.printf("Total time elapsed: %.3fs%n", (tEnd - tStart) / 1000.);
         System.out.printf("Avg. time per query: %.3fms%n", (tEnd - tStart) / ((float) nQueries));
      }

   }

   public static void main(String[] args) throws IOException {
      String indexPath = args[0];
      int dimension = Integer.parseInt(args[1]);
      int nQueries = Integer.parseInt(args[2]);
      testSpeed(indexPath, dimension, nQueries);
   }

}
