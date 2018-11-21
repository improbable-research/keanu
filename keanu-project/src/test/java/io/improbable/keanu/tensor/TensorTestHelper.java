package io.improbable.keanu.tensor;

import static org.junit.Assert.assertEquals;

public class TensorTestHelper {

    public static void doesDownRankOnSliceRank3To2(Tensor x) {
        Tensor slice = x.slice(0, 1);
        assertEquals(2, slice.getRank());
    }

    public static void doesDownRankOnSliceRank2To1(Tensor x) {
        Tensor slice = x.slice(0, 1);
        assertEquals(1, slice.getRank());
    }

    public static void doesDownRankOnSliceRank1ToScalar(Tensor x) {
        Tensor slice = x.slice(0, 1);
        assertEquals(0, slice.getRank());
    }
}
