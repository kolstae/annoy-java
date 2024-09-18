package com.spotify.annoy;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collection;
import java.util.PriorityQueue;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.roaringbitmap.RoaringBitmap;

/**
 * Read-only Approximate Nearest Neighbor Index which queries
 * databases created by annoy.
 */
public class ANNIndex implements AnnoyIndex {
   private static final CosineDistance COSINE_DISTANCE = new CosineDistance();
   private static final int INT_SIZE = 4;
   private static final int FLOAT_SIZE = 4;
   private static final ValueLayout.OfInt INT = ValueLayout.OfInt.JAVA_INT
         .withOrder(ByteOrder.LITTLE_ENDIAN);
   private static final ValueLayout.OfFloat FLOAT = ValueLayout.OfFloat.JAVA_FLOAT
         .withOrder(ByteOrder.LITTLE_ENDIAN);

   private final Collection<PQEntry> roots;

   private final int minLeafSize;
   private final int indexTypeOffset;

   // size of C structs in bytes (initialized in init)
   private final int headerSize;
   private final long nodeSize;

   private final Distance distance;
   private final long numNodes;
   private final Arena arena;
   private final MemorySegment segment;


   /**
    * Construct and load an Annoy index of a specific type (euclidean / angular).
    *
    * @param dimension dimensionality of tree, e.g. 40
    * @param filename  filename of tree
    * @param indexType type of index
    * @throws IOException if file can't be loaded
    */
   public ANNIndex(final int dimension,
                   final String filename,
                   IndexType indexType) throws IOException {
      this(dimension, filename, indexType, 0);
   }

   /**
    * Construct and load an (Angular) Annoy index.
    *
    * @param dimension dimensionality of tree, e.g. 40
    * @param filename  filename of tree
    * @throws IOException if file can't be loaded
    */
   public ANNIndex(final int dimension,
                   final String filename) throws IOException {
      this(dimension, filename, IndexType.ANGULAR);
   }

   ANNIndex(final int dimension,
            final String filename,
            IndexType indexType,
            final int blockSize) throws IOException {
      distance = indexType == IndexType.ANGULAR ? COSINE_DISTANCE : new EuclideanDistance();
      indexTypeOffset = (indexType == IndexType.ANGULAR) ? 4 : 8;
      headerSize = (indexType == IndexType.ANGULAR) ? 12 : 16;
      // we can store up to minLeafSize children in leaf nodes (we put
      // them where the separating plane normally goes)
      this.minLeafSize = dimension + 2;
      this.nodeSize = headerSize + (long) FLOAT_SIZE * dimension;
      final LoadedIndex loaded = load(filename, nodeSize);
      arena = loaded.arena;
      segment = loaded.segment;
      numNodes = loaded.numNodes;
      roots = Arrays.stream(loaded.roots)
            .mapToObj(r -> new PQEntry(1e30f, r))
            .collect(Collectors.toList());
   }

   private static LoadedIndex load(final String filename, long nodeSize) throws IOException {
      try (final RandomAccessFile file = new RandomAccessFile(filename, "r")) {
         final long fileSize = file.length();
         if (fileSize == 0L) {
            throw new IOException("Index is a 0-byte file?");
         }
         if (fileSize % nodeSize != 0) {
            throw new RuntimeException("ANNIndex initiated with wrong dimension size");
         }
         final long numNodes = fileSize / nodeSize;
         // Two valid relations between dimension and file size:
         // 1) rest % nodeSize == 0 makes sure either everything fits into buffer or rest is a multiple of nodeSize;
         // 2) (file_size - rest) % nodeSize == 0 makes sure everything else is a multiple of nodeSize.
         long position = 0;
         boolean process = true;
         int m = -1;
         long index = fileSize;
         final LongStream.Builder roots = LongStream.builder();
         try (final FileChannel fc = file.getChannel()) {
            final Arena arena = Arena.ofShared();
            final MemorySegment annBuf = fc
                  .map(FileChannel.MapMode.READ_ONLY, position, fileSize, arena);
            for (long i = fileSize - nodeSize; i >= 0; i -= nodeSize) {
               index -= nodeSize;
               int k = annBuf.get(INT, i);
               if (m == -1 || k == m) {
                  roots.add(index);
                  m = k;
               } else {
                  break;
               }
            }
            return new LoadedIndex(numNodes, arena, annBuf, roots.build().toArray());
         }
      }
   }

   private float getFloatInAnnBuf(long pos) {
      return segment.get(FLOAT, pos);
   }

   private int getIntInAnnBuf(long pos) {
      return segment.get(INT, pos);
   }

   private float getNodeBias(final long nodeOffset) { // euclidean-only
      return getFloatInAnnBuf(nodeOffset + 4);
   }

   @Override
   public final float[] getItemVector(final int itemIndex) {
      if (itemIndex < 0 || itemIndex >= numNodes) {
         throw new IndexOutOfBoundsException(String.format("[0 - %d] was: %d", numNodes - 1, itemIndex));
      }
      return getNodeVector(itemIndex * nodeSize);
   }

   private float[] getNodeVector(final long nodeOffset) {
      final long offset = nodeOffset + headerSize;
      return segment.asSlice(offset, nodeSize - headerSize).toArray(FLOAT);
   }

   private static float euclideanDistance(final float[] u, final float[] v) {
      float n = 0;
      for (int i = 0; i < u.length; i++) {
         final float d = u[i] - v[i];
         n = Math.fma(d, d, n);
      }
      return (float) Math.sqrt(n);
   }

   private static float cosineMargin(final float[] u, final float[] v) {
      float d = 0;
      float uu = 0;
      float vv = 0;
      for (int i = 0; i < u.length; i++) {
         final float ui = u[i];
         final float vi = v[i];
         d = Math.fma(ui, vi, d);
         uu = Math.fma(ui, ui, uu);
         vv = Math.fma(vi, vi, vv);
      }
      return (float) (d / (Math.sqrt(uu) * Math.sqrt(vv)));
   }

   private static float euclideanMargin(final float[] u, final float[] v, final float bias) {
      float d = bias;
      for (int i = 0; i < u.length; i++)
         d = Math.fma(u[i], v[i], d);
      return d;
   }

   /**
    * Closes this stream and releases any system resources associated
    * with it. If the stream is already closed then invoking this
    * method has no effect.
    * <p/>
    * <p> As noted in {@link AutoCloseable#close()}, cases where the
    * close may fail require careful attention. It is strongly advised
    * to relinquish the underlying resources and to internally
    * <em>mark</em> the {@code Closeable} as closed, prior to throwing
    * the {@code IOException}.
    */
   @Override
   public void close() {
      arena.close();
   }

   static final class PQEntry implements Comparable<PQEntry> {
      private final float margin;
      private final long nodeOffset;

      PQEntry(final float margin, final long nodeOffset) {
         this.margin = margin;
         this.nodeOffset = nodeOffset;
      }

      long getNodeOffset() {
         return nodeOffset;
      }

      float getMargin() {
         return margin;
      }

      @Override
      public int compareTo(final PQEntry o) {
         return Float.compare(o.margin, margin);
      }
   }

   @Override
   public final Stream<IdxAndScore> getNearest(float[] queryVector, IntPredicate pred, int nResults) {
      final PriorityQueue<PQEntry> pq = new PriorityQueue<>(roots.size() * FLOAT_SIZE);
      pq.addAll(roots);

      final int maxSize = roots.size() * nResults;
      final RoaringBitmap bm = new RoaringBitmap();
      final TopNBuilder builder = new TopNBuilder(nResults);
      int seen = 0;
      while (seen < maxSize && !pq.isEmpty()) {
         final long topNodeOffset = pq.poll().nodeOffset;
         final int nDescendants = getIntInAnnBuf(topNodeOffset);
         if (nDescendants == 1) {  // n_descendants
            final int idx = (int) (topNodeOffset / nodeSize);
            if (bm.checkedAdd(idx)) {
               if (pred.test(idx)) {
                  builder.add(idx, distance.distance(getNodeVector(topNodeOffset), queryVector));
               }
               seen++;
            }
         } else if (nDescendants <= minLeafSize) {
            for (int i = 0; i < nDescendants; i++) {
               final int idx = getIntInAnnBuf(topNodeOffset + indexTypeOffset + i * INT_SIZE);
               if (bm.checkedAdd(idx)) {
                  if (pred.test(idx)) {
                     builder.add(idx, distance.distance(getItemVector(idx), queryVector));
                  }
                  seen++;
               }
            }
         } else {
            final float margin = distance.margin(getNodeVector(topNodeOffset), queryVector, topNodeOffset);
            final long childrenMemOffset = topNodeOffset + indexTypeOffset;
            pq.add(new PQEntry(-margin, nodeSize * getIntInAnnBuf(childrenMemOffset)));
            pq.add(new PQEntry(margin, nodeSize * getIntInAnnBuf(childrenMemOffset + INT_SIZE)));
         }
      }

      return builder.stream().map(e -> new IdxAndScore((int) e.nodeOffset, e.margin));
   }


   /**
    * a test query program.
    *
    * @param args tree filename, dimension, indextype ("angular" or
    *             "euclidean" and query item id.
    * @throws IOException if unable to load index
    */
   public static void main(final String[] args) throws IOException {

      String indexPath = args[0];                 // 0
      int dimension = Integer.parseInt(args[1]);  // 1
      IndexType indexType;                        // 2
      if ("angular".equalsIgnoreCase(args[2]))
         indexType = IndexType.ANGULAR;
      else if ("euclidean".equalsIgnoreCase(args[2]))
         indexType = IndexType.EUCLIDEAN;
      else throw new RuntimeException("wrong index type specified");
      int queryItem = Integer.parseInt(args[3]);  // 3

      try (ANNIndex annIndex = new ANNIndex(dimension, indexPath, indexType);) {

         // input vector
         float[] u = annIndex.getItemVector(queryItem);
         System.out.printf("vector[%d]: ", queryItem);
         for (float x : u) {
            System.out.printf("%2.2f ", x);
         }
         System.out.printf("\n");

         annIndex.getNearest(queryItem, i -> true, 10)
               .mapToInt(IdxAndScore::idx)
               .forEach(nn -> System.out.printf("%d %d %f\n", queryItem, nn, annIndex.distance.distance(u, annIndex.getItemVector(nn))));
      }
   }

   private interface Distance {
      float distance(float[] u, float[] v);

      float margin(float[] u, float[] v, long nodeOffset);
   }

   private static final class CosineDistance implements Distance {
      @Override
      public float distance(float[] u, float[] v) {
         return cosineMargin(u, v);
      }

      @Override
      public float margin(float[] u, float[] v, long nodeOffset) {
         return distance(u, v);
      }
   }

   private final class EuclideanDistance implements Distance {
      @Override
      public float distance(float[] u, float[] v) {
         return 1 - euclideanDistance(u, v);
      }

      @Override
      public float margin(float[] u, float[] v, long nodeOffset) {
         return euclideanMargin(u, v, getNodeBias(nodeOffset));
      }
   }

   private static final class LoadedIndex {
      private final long numNodes;
      private final Arena arena;
      private final MemorySegment segment;
      private final long[] roots;

      LoadedIndex(long numNodes, Arena arena, MemorySegment segment, long[] roots) {
         this.numNodes = numNodes;
         this.arena = arena;
         this.segment = segment;
         this.roots = roots;
      }
   }
}
