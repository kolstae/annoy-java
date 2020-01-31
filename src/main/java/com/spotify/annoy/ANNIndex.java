package com.spotify.annoy;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collection;
import java.util.PriorityQueue;
import java.util.function.Consumer;
import java.util.function.IntPredicate;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.roaringbitmap.RoaringBitmap;
import sun.misc.Unsafe;

/**
 * Read-only Approximate Nearest Neighbor Index which queries
 * databases created by annoy.
 */
public class ANNIndex implements AnnoyIndex {
  private static final CosineDistance COSINE_DISTANCE = new CosineDistance();
  private static final int INT_SIZE = 4;
  private static final int FLOAT_SIZE = 4;

  private final Collection<PQEntry> roots;

  private final int dimension;
  private final int minLeafSize;
  private final int indexTypeOffset;

  // size of C structs in bytes (initialized in init)
  private final int headerSize;
  private final long nodeSize;

  private final int blockSize;
  private final Distance distance;
  private final ByteBuffer[] buffers;
  private final int numNodes;


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
    this.dimension = dimension;
    distance = indexType == IndexType.ANGULAR ? COSINE_DISTANCE : new EuclideanDistance();
    indexTypeOffset = (indexType == IndexType.ANGULAR) ? 4 : 8;
    headerSize = (indexType == IndexType.ANGULAR) ? 12 : 16;
    // we can store up to minLeafSize children in leaf nodes (we put
    // them where the separating plane normally goes)
    this.minLeafSize = this.dimension + 2;
    this.nodeSize = headerSize + FLOAT_SIZE * this.dimension;
    final int maxNodesInBuffer = Math.toIntExact(blockSize == 0
          ? Integer.MAX_VALUE / nodeSize
          : blockSize * nodeSize);
    this.blockSize = Math.toIntExact(maxNodesInBuffer * nodeSize);
    final LoadedIndex loaded = load(filename, nodeSize, maxNodesInBuffer, this.blockSize);
    buffers = loaded.buffers;
    numNodes = loaded.numNodes;
    roots = Arrays.stream(loaded.roots)
          .mapToObj(r -> new PQEntry(1e30f, r))
          .collect(Collectors.toList());
  }

  private static LoadedIndex load(final String filename, long nodeSize, int maxNodesInBuffer, int blockSize) throws IOException {
    try (final RandomAccessFile file = new RandomAccessFile(filename, "r")) {
        final long fileSize = file.length();
        if (fileSize == 0L) {
            throw new IOException("Index is a 0-byte file?");
        }

        final int numNodes = Math.toIntExact(fileSize / nodeSize);
        int buffIndex = (numNodes - 1) / maxNodesInBuffer;
        final int rest = Math.toIntExact(fileSize % blockSize);
        int lastBlockSize = (rest > 0 ? rest : blockSize);
        // Two valid relations between dimension and file size:
        // 1) rest % nodeSize == 0 makes sure either everything fits into buffer or rest is a multiple of nodeSize;
        // 2) (file_size - rest) % nodeSize == 0 makes sure everything else is a multiple of nodeSize.
        if (rest % nodeSize != 0 || (fileSize - rest) % nodeSize != 0) {
            throw new RuntimeException("ANNIndex initiated with wrong dimension size");
        }
        long position = fileSize - lastBlockSize;
        final ByteBuffer[] buffers = new ByteBuffer[buffIndex + 1];
        boolean process = true;
        int m = -1;
        long index = fileSize;
        final LongStream.Builder roots = LongStream.builder();
        try (final FileChannel fc = file.getChannel()) {
          while (position >= 0) {
            final ByteBuffer annBuf = fc
                    .map(FileChannel.MapMode.READ_ONLY, position, lastBlockSize)
                    .order(ByteOrder.LITTLE_ENDIAN);

            buffers[buffIndex--] = annBuf;

            for (int i = lastBlockSize - (int) nodeSize; process && i >= 0; i -= nodeSize) {
              index -= nodeSize;
              int k = annBuf.getInt(i);  // node[i].n_descendants
              if (m == -1 || k == m) {
                roots.add(index);
                m = k;
              } else {
                process = false;
              }
            }
            lastBlockSize = blockSize;
            position -= lastBlockSize;
            }
        }
        return new LoadedIndex(numNodes, buffers, roots.build().toArray());
    }
  }

  private float getFloatInAnnBuf(long pos) {
    int b = (int) (pos / blockSize);
    int f = (int) (pos % blockSize);
    return buffers[b].getFloat(f);
  }

  private int getIntInAnnBuf(long pos) {
    int b = (int) (pos / blockSize);
    int i = (int) (pos % blockSize);
    return buffers[b].getInt(i);
  }

  private void getNodeVector(final long nodeOffset, float[] v) {
    final ByteBuffer nodeBuf = buffers[(int) (nodeOffset / blockSize)];
    final int offset = (int) ((nodeOffset % blockSize) + headerSize);
    for (int i = 0; i < dimension; i++) {
      v[i] = nodeBuf.getFloat(offset + i * FLOAT_SIZE);
    }
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
    float[] v = new float[dimension];
    getNodeVector(nodeOffset, v);
    return v;
  }

  private static float euclideanDistance(final float[] u, final float[] v) {
    double n = 0;
    for (int i = 0; i < u.length; i++) {
      final float d = u[i] - v[i];
      n += d * d;
    }
    return (float) Math.sqrt(n);
  }

  private static float cosineMargin(final float[] u, final float[] v) {
    double d = 0;
    double uu = 0;
    double vv = 0;
    for (int i = 0; i < u.length; i++) {
      final float ui = u[i];
      final float vi = v[i];
      d += ui * vi;
      uu += ui * ui;
      vv += vi * vi;
    }
    return (float) (d / (Math.sqrt(uu) * Math.sqrt(vv)));
  }

  private static float euclideanMargin(final float[] u, final float[] v, final float bias) {
    float d = bias;
    for (int i = 0; i < u.length; i++)
      d += u[i] * v[i];
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
    for (int i = 0; i < buffers.length; i++) {
      final ByteBuffer buffer = buffers[i];
      buffers[i] = ByteBuffer.allocate(0);
      forceClose(buffer);
    }
  }

  private static final Consumer<ByteBuffer> cleaner;

  static {
    try {
      final Field f = Unsafe.class.getDeclaredField("theUnsafe");
      f.setAccessible(true);
      final Unsafe unsafe = (Unsafe) f.get(null);
      cleaner = unsafe::invokeCleaner;
    } catch (NoSuchFieldException | IllegalAccessException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static void forceClose(ByteBuffer b) {
    try {
      if (b.isDirect()) {
          cleaner.accept(b);
      }
    } catch (Throwable e) {
      e.printStackTrace();
    }
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

    ANNIndex annIndex = new ANNIndex(dimension, indexPath, indexType);

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
    private final int numNodes;
    private final ByteBuffer[] buffers;
    private final long[] roots;

    LoadedIndex(int numNodes, ByteBuffer[] buffers, long[] roots) {
        this.numNodes = numNodes;
      this.buffers = buffers;
      this.roots = roots;
    }
  }
}
