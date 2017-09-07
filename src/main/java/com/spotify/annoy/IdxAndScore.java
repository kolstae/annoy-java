package com.spotify.annoy;

public final class IdxAndScore {
   private final int idx;
   private final double score;

   public IdxAndScore(int idx, double score) {
      this.idx = idx;
      this.score = score;
   }

   public int idx() {
      return idx;
   }

   public double score() {
      return score;
   }
}
