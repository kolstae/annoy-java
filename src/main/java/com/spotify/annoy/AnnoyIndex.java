package com.spotify.annoy;

import java.io.Closeable;
import java.util.function.IntPredicate;
import java.util.stream.Stream;

/**
 * AnnoyIndex interface, provided to aid with dependency injection in tests.
 */
public interface AnnoyIndex extends Closeable {
  /**
   * Get the vector for a given item in the tree.
   * @param itemIndex  item idx
   * @return item vector
   */
  float[] getItemVector(int itemIndex);

  /**
   * Look up nearest neighbors in the tree.
   * @param queryVector  find nearest neighbors for this query point
   * @param nResults     number of items to return
   * @return             list of items in descending nearness to query point
   */
  Stream<IdxAndScore> getNearest(float[] queryVector, IntPredicate pred, int nResults);

  /**
   * Look up nearest neighbors in the tree.
   * @param itemIndex    find nearest neighbors for this item
   * @param nResults     number of items to return
   * @return             list of items in descending nearness to query point
   */
  default Stream<IdxAndScore> getNearest(int itemIndex, IntPredicate pred, int nResults) {
    return getNearest(getItemVector(itemIndex), pred, nResults);
  }
}
