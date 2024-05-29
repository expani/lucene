/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.util.bkd;

import org.apache.lucene.index.PointValues.IntersectVisitor;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.DocBaseBitSetIterator;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.util.Arrays;

final class DocIdsWriter {

  private static final byte CONTINUOUS_IDS = (byte) -2;
  private static final byte BITSET_IDS = (byte) -1;
  private static final byte DELTA_BPV_16 = (byte) 16;
  private static final byte BPV_24 = (byte) 24;

  private static final byte BPV_PREFIX = (byte) -23;
  private static final byte BPV_32 = (byte) 32;
  // These signs are legacy, should no longer be used in the writing side.
  private static final byte LEGACY_DELTA_VINT = (byte) 0;

  private final int[] scratch, scratch2;

  DocIdsWriter(int maxPointsInLeaf) {
    scratch = new int[maxPointsInLeaf];
    scratch2 = new int[maxPointsInLeaf];
  }

  void writeDocIds(int[] docIds, int start, int count, DataOutput out) throws IOException {
    // docs can be sorted either when all docs in a block have the same value
    // or when a segment is sorted
    boolean strictlySorted = true;
    int min = docIds[0];
    int max = docIds[0];
    int overallCommonBits = Integer.MAX_VALUE;
    for (int i = 1; i < count; ++i) {
      int last = docIds[start + i - 1];
      int current = docIds[start + i];
      int currCommon = commonPrefixBits(current, last);
      overallCommonBits = Math.min(overallCommonBits, currCommon);
      if (last >= current) {
        strictlySorted = false;
      }
      min = Math.min(min, current);
      max = Math.max(max, current);
    }

    int min2max = max - min + 1;
    if (strictlySorted) {
      if (min2max == count) {
        // continuous ids, typically happens when segment is sorted
        out.writeByte(CONTINUOUS_IDS);
        out.writeVInt(docIds[start]);
        return;
      } else if (min2max <= (count << 4)) {
        assert min2max > count : "min2max: " + min2max + ", count: " + count;
        // Only trigger bitset optimization when max - min + 1 <= 16 * count in order to avoid
        // expanding too much storage.
        // A field with lower cardinality will have higher probability to trigger this optimization.
        out.writeByte(BITSET_IDS);
        writeIdsAsBitSet(docIds, start, count, out);
        return;
      }
    }

    if (min2max <= 0xFFFF) {
      out.writeByte(DELTA_BPV_16);
      for (int i = 0; i < count; i++) {
        scratch[i] = docIds[start + i] - min;
      }
      out.writeVInt(min);
      final int halfLen = count >>> 1;
      for (int i = 0; i < halfLen; ++i) {
        scratch[i] = scratch[halfLen + i] | (scratch[i] << 16);
      }
      for (int i = 0; i < halfLen; i++) {
        out.writeInt(scratch[i]);
      }
      if ((count & 1) == 1) {
        out.writeShort((short) scratch[count - 1]);
      }
    } else {
      if (max <= 0xFFFFFF) {
        if (overallCommonBits <= 8) {
          out.writeByte(BPV_24);
          // write them the same way we are reading them.
          int i;
          for (i = 0; i < count - 7; i += 8) {
            int doc1 = docIds[start + i];
            int doc2 = docIds[start + i + 1];
            int doc3 = docIds[start + i + 2];
            int doc4 = docIds[start + i + 3];
            int doc5 = docIds[start + i + 4];
            int doc6 = docIds[start + i + 5];
            int doc7 = docIds[start + i + 6];
            int doc8 = docIds[start + i + 7];
            long l1 = (doc1 & 0xffffffL) << 40 | (doc2 & 0xffffffL) << 16 | ((doc3 >>> 8) & 0xffffL);
            long l2 =
                    (doc3 & 0xffL) << 56
                            | (doc4 & 0xffffffL) << 32
                            | (doc5 & 0xffffffL) << 8
                            | ((doc6 >> 16) & 0xffL);
            long l3 = (doc6 & 0xffffL) << 48 | (doc7 & 0xffffffL) << 24 | (doc8 & 0xffffffL);
            out.writeLong(l1);
            out.writeLong(l2);
            out.writeLong(l3);
          }
          for (; i < count; ++i) {
            out.writeShort((short) (docIds[start + i] >>> 8));
            out.writeByte((byte) docIds[start + i]);
          }
        } else {
          //System.out.println("Writing 24 bits docIds prefix with len as " + overallCommonBits);
          writeIdsWithCommonPrefix(docIds, start, count, out, overallCommonBits);
        }
      } else {
        if (overallCommonBits < 1) {
          out.writeByte(BPV_32);
          for (int i = 0; i < count; i++) {
            out.writeInt(docIds[start + i]);
          }
        } else {
          //System.out.println("Writing 32 bits docIds prefix with len as " + overallCommonBits);
          writeIdsWithCommonPrefix(docIds, start, count, out, overallCommonBits);
        }
      }
    }
  }

  private void writeIdsWithCommonPrefix(int[] docIds, int start, int count, DataOutput out,
                                               int totalCommonPrefixBits) throws IOException {

    out.writeByte(BPV_PREFIX);

    int uncommonBits = 32 - totalCommonPrefixBits;

    // Will use less bits if kept at LSB
    int commonPrefixLSB = docIds[0] >>> uncommonBits;

    out.writeVInt(totalCommonPrefixBits);
    out.writeVInt(commonPrefixLSB);

    int numIntsRequiredForPacking = (uncommonBits * count) / 32;
    numIntsRequiredForPacking += ((uncommonBits * count) % 32 == 0) ? 0 : 1;

    int packedIntPtr = 0;
    int usedBits = 0;
    scratch[packedIntPtr] = 0;

    for (int i = start; i < (start + count); i++) {
      int prefixRemovedDocId = (docIds[i] << totalCommonPrefixBits) >>> totalCommonPrefixBits;
      int freeBits = 32 - usedBits;
      if (freeBits >= uncommonBits) {
        scratch[packedIntPtr] = (prefixRemovedDocId << (32 - (uncommonBits + usedBits))) | scratch[packedIntPtr];
        usedBits += uncommonBits;
        if (usedBits == 32) {
          packedIntPtr++;
          scratch[packedIntPtr] = 0;
          usedBits = 0;
        }
      } else {
        scratch[packedIntPtr] = (prefixRemovedDocId >>> (uncommonBits - freeBits)) | scratch[packedIntPtr];
        packedIntPtr++;
        scratch[packedIntPtr] = 0;
        scratch[packedIntPtr] = (prefixRemovedDocId << (totalCommonPrefixBits + freeBits)) | scratch[packedIntPtr];
        usedBits = uncommonBits - freeBits;
      }
    }

    for (int i = 0; i < numIntsRequiredForPacking; i++) {
      out.writeInt(scratch[i]);
    }

//    System.out.printf("\nWritten leaf block having start %s with %s docIds and common prefix length %s as %s\n", start, count, totalCommonPrefixBits, numIntsRequiredForPacking);
//
//    int[] decompressArr = readIntsPrefixFromArr(scratch, count, totalCommonPrefixBits, commonPrefixLSB);
//    boolean res = Arrays.equals(Arrays.copyOfRange(docIds, start, start + count), decompressArr);
//
//    if (!res) {
//      System.out.printf("\nMismatch detected !!! Actual DocIds %s Decompressed DocIds %s Compressed Arr %s \n",
//              Arrays.toString(docIds),
//              Arrays.toString(decompressArr),
//              Arrays.toString(Arrays.copyOfRange(scratch, 0, numIntsRequiredForPacking)));
//    }

  }

  private static void writeIdsAsBitSet(int[] docIds, int start, int count, DataOutput out)
      throws IOException {
    int min = docIds[start];
    int max = docIds[start + count - 1];

    final int offsetWords = min >> 6;
    final int offsetBits = offsetWords << 6;
    final int totalWordCount = FixedBitSet.bits2words(max - offsetBits + 1);
    long currentWord = 0;
    int currentWordIndex = 0;

    out.writeVInt(offsetWords);
    out.writeVInt(totalWordCount);
    // build bit set streaming
    for (int i = 0; i < count; i++) {
      final int index = docIds[start + i] - offsetBits;
      final int nextWordIndex = index >> 6;
      assert currentWordIndex <= nextWordIndex;
      if (currentWordIndex < nextWordIndex) {
        out.writeLong(currentWord);
        currentWord = 0L;
        currentWordIndex++;
        while (currentWordIndex < nextWordIndex) {
          currentWordIndex++;
          out.writeLong(0L);
        }
      }
      currentWord |= 1L << index;
    }
    out.writeLong(currentWord);
    assert currentWordIndex + 1 == totalWordCount;
  }

  /** Read {@code count} integers into {@code docIDs}. */
  void readInts(IndexInput in, int count, int[] docIDs) throws IOException {
    final int bpv = in.readByte();
    switch (bpv) {
      case CONTINUOUS_IDS:
        readContinuousIds(in, count, docIDs);
        break;
      case BITSET_IDS:
        readBitSet(in, count, docIDs);
        break;
      case DELTA_BPV_16:
        readDelta16(in, count, docIDs);
        break;
      case BPV_24:
        readInts24(in, count, docIDs);
        break;
      case BPV_32:
        readInts32(in, count, docIDs);
        break;
      case LEGACY_DELTA_VINT:
        readLegacyDeltaVInts(in, count, docIDs);
        break;
      case BPV_PREFIX:
        readIntsPrefix(in, count, docIDs);
        break;
      default:
        throw new IOException("Unsupported number of bits per value: " + bpv);
    }
  }

  private static DocIdSetIterator readBitSetIterator(IndexInput in, int count) throws IOException {
    int offsetWords = in.readVInt();
    int longLen = in.readVInt();
    long[] bits = new long[longLen];
    in.readLongs(bits, 0, longLen);
    FixedBitSet bitSet = new FixedBitSet(bits, longLen << 6);
    return new DocBaseBitSetIterator(bitSet, count, offsetWords << 6);
  }

  private static void readContinuousIds(IndexInput in, int count, int[] docIDs) throws IOException {
    int start = in.readVInt();
    for (int i = 0; i < count; i++) {
      docIDs[i] = start + i;
    }
  }

  private static void readLegacyDeltaVInts(IndexInput in, int count, int[] docIDs)
      throws IOException {
    int doc = 0;
    for (int i = 0; i < count; i++) {
      doc += in.readVInt();
      docIDs[i] = doc;
    }
  }

  private static void readBitSet(IndexInput in, int count, int[] docIDs) throws IOException {
    DocIdSetIterator iterator = readBitSetIterator(in, count);
    int docId, pos = 0;
    while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
      docIDs[pos++] = docId;
    }
    assert pos == count : "pos: " + pos + ", count: " + count;
  }

  private static void readDelta16(IndexInput in, int count, int[] docIDs) throws IOException {
    final int min = in.readVInt();
    final int halfLen = count >>> 1;
    in.readInts(docIDs, 0, halfLen);
    for (int i = 0; i < halfLen; ++i) {
      int l = docIDs[i];
      docIDs[i] = (l >>> 16) + min;
      docIDs[halfLen + i] = (l & 0xFFFF) + min;
    }
    if ((count & 1) == 1) {
      docIDs[count - 1] = Short.toUnsignedInt(in.readShort()) + min;
    }
  }

  private static void readInts24(IndexInput in, int count, int[] docIDs) throws IOException {
    int i;
    for (i = 0; i < count - 7; i += 8) {
      long l1 = in.readLong();
      long l2 = in.readLong();
      long l3 = in.readLong();
      docIDs[i] = (int) (l1 >>> 40);
      docIDs[i + 1] = (int) (l1 >>> 16) & 0xffffff;
      docIDs[i + 2] = (int) (((l1 & 0xffff) << 8) | (l2 >>> 56));
      docIDs[i + 3] = (int) (l2 >>> 32) & 0xffffff;
      docIDs[i + 4] = (int) (l2 >>> 8) & 0xffffff;
      docIDs[i + 5] = (int) (((l2 & 0xff) << 16) | (l3 >>> 48));
      docIDs[i + 6] = (int) (l3 >>> 24) & 0xffffff;
      docIDs[i + 7] = (int) l3 & 0xffffff;
    }
    for (; i < count; ++i) {
      docIDs[i] = (Short.toUnsignedInt(in.readShort()) << 8) | Byte.toUnsignedInt(in.readByte());
    }
  }

  private static void readInts32(IndexInput in, int count, int[] docIDs) throws IOException {
    in.readInts(docIDs, 0, count);
  }

  /**
   * Read {@code count} integers and feed the result directly to {@link
   * IntersectVisitor#visit(int)}.
   */
  void readInts(IndexInput in, int count, IntersectVisitor visitor) throws IOException {
    final int bpv = in.readByte();
    switch (bpv) {
      case CONTINUOUS_IDS:
        readContinuousIds(in, count, visitor);
        break;
      case BITSET_IDS:
        readBitSet(in, count, visitor);
        break;
      case DELTA_BPV_16:
        readDelta16(in, count, visitor);
        break;
      case BPV_24:
        readInts24(in, count, visitor);
        break;
      case BPV_32:
        readInts32(in, count, visitor);
        break;
      case LEGACY_DELTA_VINT:
        readLegacyDeltaVInts(in, count, visitor);
        break;
      case BPV_PREFIX:
        readIntsPrefix(in, count, visitor);
        break;
      default:
        throw new IOException("Unsupported number of bits per value: " + bpv);
    }
  }

  private static void readBitSet(IndexInput in, int count, IntersectVisitor visitor)
      throws IOException {
    DocIdSetIterator bitSetIterator = readBitSetIterator(in, count);
    visitor.visit(bitSetIterator);
  }

  private static void readContinuousIds(IndexInput in, int count, IntersectVisitor visitor)
      throws IOException {
    int start = in.readVInt();
    int extra = start & 63;
    int offset = start - extra;
    int numBits = count + extra;
    FixedBitSet bitSet = new FixedBitSet(numBits);
    bitSet.set(extra, numBits);
    visitor.visit(new DocBaseBitSetIterator(bitSet, count, offset));
  }

  private static void readLegacyDeltaVInts(IndexInput in, int count, IntersectVisitor visitor)
      throws IOException {
    int doc = 0;
    for (int i = 0; i < count; i++) {
      doc += in.readVInt();
      visitor.visit(doc);
    }
  }

  private void readDelta16(IndexInput in, int count, IntersectVisitor visitor) throws IOException {
    readDelta16(in, count, scratch);
    for (int i = 0; i < count; i++) {
      visitor.visit(scratch[i]);
    }
  }

  private static void readInts24(IndexInput in, int count, IntersectVisitor visitor)
      throws IOException {
    int i;
    for (i = 0; i < count - 7; i += 8) {
      long l1 = in.readLong();
      long l2 = in.readLong();
      long l3 = in.readLong();
      visitor.visit((int) (l1 >>> 40));
      visitor.visit((int) (l1 >>> 16) & 0xffffff);
      visitor.visit((int) (((l1 & 0xffff) << 8) | (l2 >>> 56)));
      visitor.visit((int) (l2 >>> 32) & 0xffffff);
      visitor.visit((int) (l2 >>> 8) & 0xffffff);
      visitor.visit((int) (((l2 & 0xff) << 16) | (l3 >>> 48)));
      visitor.visit((int) (l3 >>> 24) & 0xffffff);
      visitor.visit((int) l3 & 0xffffff);
    }
    for (; i < count; ++i) {
      visitor.visit((Short.toUnsignedInt(in.readShort()) << 8) | Byte.toUnsignedInt(in.readByte()));
    }
  }

  private void readInts32(IndexInput in, int count, IntersectVisitor visitor) throws IOException {
    in.readInts(scratch, 0, count);
    for (int i = 0; i < count; i++) {
      visitor.visit(scratch[i]);
    }
  }

  private void readIntsPrefix(IndexInput in, int count, int[] docIds) throws IOException {

    if (count == 0) {
      return;
    }

    int commonPrefixLength = in.readVInt();
    int commonPrefix = in.readVInt();

    int uncommonBits = 32 - commonPrefixLength;

    int numIntsRequiredForPacking = (uncommonBits * count) / 32;
    numIntsRequiredForPacking += ((uncommonBits * count) % 32 == 0) ? 0 : 1;

    in.readInts(scratch, 0, numIntsRequiredForPacking);

    // Common Prefix is stored as by shifting fully towards LSB, but during reconstruction we need it shifted towards MSB
    int commonPrefixMSB = commonPrefix << uncommonBits;

    //System.out.printf("\nReading docIds having commonPrefix Len as %s and count as %s\n", commonPrefixLength, count);

    int scratchPtr = 0;
    int currentPackedInt = scratch[scratchPtr];
    int availableBits = 32;
    int totalIntsRead = 1;

    for (int i = 0; i < count; i++) {
      if (availableBits >= uncommonBits) {
        int docIdUncommon = ((currentPackedInt >>> (availableBits - uncommonBits)) << commonPrefixLength) >>> commonPrefixLength;
        docIds[i] = commonPrefixMSB | docIdUncommon;
        availableBits -= uncommonBits;
        if (availableBits == 0 && i < (count - 1)) {
          currentPackedInt = scratch[++scratchPtr];
          totalIntsRead++;
          availableBits = 32;
        }
      } else {
        int part1DocId = currentPackedInt << (32 - availableBits);
        int bitsLeftToRead = uncommonBits - availableBits;

        currentPackedInt = scratch[++scratchPtr];
        totalIntsRead++;
        int part2DocId = (currentPackedInt >>> (32 - bitsLeftToRead)) << (32 - (bitsLeftToRead + availableBits));

        int uncommonDocId = (part1DocId | part2DocId) >>> commonPrefixLength;
        docIds[i] = commonPrefixMSB | uncommonDocId;
        availableBits = 32 - bitsLeftToRead;
      }
    }

//    System.out.printf("\nTotal Ints read for %s docIds is %s having commonPrefix Len as %s with availableBits %s Array read %s\n",
//            count, totalIntsRead, commonPrefixLength, availableBits, Arrays.toString(docIds));

  }

  private int[] readIntsPrefixFromArr(int[] packedDocIds, int count, int commonPrefixLength, int commonPrefix) {

    if (count == 0) {
      return new int[0];
    }

    int[] docIds = new int[count];

    int uncommonBits = 32 - commonPrefixLength;
    // Common Prefix is stored as by shifting fully towards LSB, but during reconstruction we need it shifted towards MSB
    int commonPrefixMSB = commonPrefix << uncommonBits;

    System.out.printf("\nReading docIds having commonPrefix Len as %s and count as %s\n", commonPrefixLength, count);

    int prefixArrPtr = 0;
    int currentPackedInt = packedDocIds[prefixArrPtr++];
    int availableBits = 32;
    int totalIntsRead = 1;

    for (int i = 0; i < count; i++) {
      if (availableBits == 0) {
        currentPackedInt = packedDocIds[prefixArrPtr++];
        totalIntsRead++;
        availableBits = 32;
      }
      if (availableBits >= uncommonBits) {
        int docIdUncommon = ((currentPackedInt >>> (availableBits - uncommonBits)) << commonPrefixLength) >>> commonPrefixLength;
        docIds[i] = commonPrefixMSB | docIdUncommon;
        availableBits -= uncommonBits;
      } else {
        int part1DocId = currentPackedInt << (32 - availableBits);
        int bitsLeftToRead = uncommonBits - availableBits;

        currentPackedInt = packedDocIds[prefixArrPtr++];
        totalIntsRead++;
        int part2DocId = (currentPackedInt >>> (32 - bitsLeftToRead)) << (32 - (bitsLeftToRead + availableBits));

        int uncommonDocId = (part1DocId | part2DocId) >>> commonPrefixLength;
        docIds[i] = commonPrefixMSB | uncommonDocId;
        availableBits = 32 - bitsLeftToRead;
      }
    }

    System.out.printf("\nTotal Ints read for checking is %s docIds is %s having commonPrefix Len as %s prefixArrPtr %s\n", count, totalIntsRead, commonPrefixLength, prefixArrPtr);

    return docIds;

  }

  private void readIntsPrefix(IndexInput in, int count, IntersectVisitor visitor) throws IOException {
     readIntsPrefix(in, count, scratch2);
    for (int i = 0; i < count; i++) {
      visitor.visit(scratch2[i]);
    }
  }

  public static int commonPrefixBits(int a, int b) {
    int commonBits = 0;

    while (a != 0 || b != 0) {
      int aBit = a & 0x80000000;
      int bBit = b & 0x80000000;

      if (aBit == bBit) {
        commonBits++;
      } else {
        break;
      }
      a = a << 1;
      b = b << 1;
    }

    return commonBits;
  }

}
