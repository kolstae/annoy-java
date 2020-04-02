package com.spotify.annoy;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.spotify.annoy.ANNIndex.PQEntry;

final class TopNBuilder {
    private PQEntry[] items;
    private int count = 0;
    private double minScore = Double.NEGATIVE_INFINITY;

    TopNBuilder(int size) {
        items = new PQEntry[size];
    }

    private boolean isNotFull() {
        return count < items.length;
    }

    Stream<PQEntry> stream() {
        return Arrays.stream(items, 0, count);
    }

    void add(int id, float score) {
        if (Float.isFinite(score)) {
          if (count == 0) {
            items[0] = new PQEntry(score, id);
            count = 1;
            minScore = score;
          } else if (score > minScore || isNotFull()) {
            count = Math.min(items.length, count + 1);
            final PQEntry item = new PQEntry(score, id);
            final int ip = Arrays.binarySearch(items, 0, count - 1, item);
            // If found put it in after the existing one - less to copy
            final int idx = ip < 0 ? -1 - ip : ip + 1;
            System.arraycopy(items, idx, items, idx + 1, count - idx - 1);
            items[idx] = item;
            minScore = items[count - 1].getMargin();
          }
        }
    }

    @Override
    public String toString() {
        return Arrays.stream(items, 0, count)
                .map(r -> "[" + r.getNodeOffset() + "] " + r.getMargin())
                .collect(Collectors.joining(", "));
    }
}
