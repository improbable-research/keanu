package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.jvm.Slicer;
import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class SlicerTest {

    @Test
    public void canParseSingleDimensionFullyDefined() {
        assertThat(Slicer.fromString("::"),
            equalTo(Slicer.builder()
                .all()
                .build()
            )
        );

        assertThat(Slicer.fromString("..."),
            equalTo(Slicer.builder()
                .ellipsis()
                .build()
            )
        );
    }

    @Test
    public void canParseSingleDimensionWithOnlyStart() {
        assertThat(Slicer.fromString("2::"),
            equalTo(Slicer.builder()
                .slice(2, null, null)
                .build()
            )
        );

        assertThat(Slicer.fromString("2:"),
            equalTo(Slicer.builder()
                .slice(2, null)
                .build()
            )
        );

        assertThat(Slicer.fromString("2"),
            equalTo(Slicer.builder()
                .slice(2)
                .build()
            )
        );
    }

    @Test
    public void canParseSingleDimensionWithOnlyStop() {

        assertThat(Slicer.fromString(":2:"),
            equalTo(Slicer.builder()
                .slice(null, 2, null)
                .build()
            )
        );

        assertThat(Slicer.fromString(":2"),
            equalTo(Slicer.builder()
                .slice(null, 2)
                .build()
            )
        );
    }

    @Test
    public void canParseSingleDimensionWithOnlyStep() {
        assertThat(Slicer.fromString("::2"),
            equalTo(Slicer.builder()
                .slice(null, null, 2)
                .build()
            )
        );
    }

    @Test
    public void canParseSingleDimensionWithStartAndStop() {
        assertThat(Slicer.fromString("2:3"),
            equalTo(Slicer.builder()
                .slice(2, 3)
                .build()
            )
        );

        assertThat(Slicer.fromString("2:3:"),
            equalTo(Slicer.builder()
                .slice(2, 3)
                .build()
            )
        );
    }

    @Test
    public void canParseSingleDimensionWithStepAndStop() {
        assertThat(Slicer.fromString(":2:3"),
            equalTo(Slicer.builder()
                .slice(null, 2, 3)
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDims() {

        assertThat(Slicer.fromString("2:3:4,::"),
            equalTo(Slicer.builder()
                .slice(2, 3, 4)
                .all()
                .build()
            )
        );

        assertThat(Slicer.fromString("2:3:4,..."),
            equalTo(Slicer.builder()
                .slice(2, 3, 4)
                .ellipsis()
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDimsAndStarts() {
        assertThat(Slicer.fromString("2,3"),
            equalTo(Slicer.builder()
                .slice(2)
                .slice(3)
                .build()
            )
        );

        assertThat(Slicer.fromString("2::,3:"),
            equalTo(Slicer.builder()
                .slice(2, null, null)
                .slice(3, null, null)
                .build()
            )
        );

        assertThat(Slicer.fromString("2:,3"),
            equalTo(Slicer.builder()
                .slice(2, null)
                .slice(3)
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDimsAndStartsAndStops() {
        assertThat(Slicer.fromString("2:3,4:5"),
            equalTo(Slicer.builder()
                .slice(2, 3)
                .slice(4, 5)
                .build()
            )
        );

        assertThat(Slicer.fromString("2:3:,4:5"),
            equalTo(Slicer.builder()
                .slice(2, 3)
                .slice(4, 5)
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDimsAndStartsAndStopsAndStep() {
        assertThat(Slicer.fromString("2:3:6,4:5:7"),
            equalTo(Slicer.builder()
                .slice(2, 3, 6)
                .slice(4, 5, 7)
                .build()
            )
        );

        assertThat(Slicer.fromString("2:3:6,...,4:5:7"),
            equalTo(Slicer.builder()
                .slice(2, 3, 6)
                .ellipsis()
                .slice(4, 5, 7)
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDimsAndStartsAndStep() {
        assertThat(Slicer.fromString("2::6,4::7"),
            equalTo(Slicer.builder()
                .slice(2, null, 6)
                .slice(4, null, 7)
                .build()
            )
        );
    }

    @Test
    public void canParseWithTwoDimsAndWhiteSpace() {
        assertThat(Slicer.fromString(" 2, 3"),
            equalTo(Slicer.builder()
                .slice(2)
                .slice(3)
                .build()
            )
        );

        assertThat(Slicer.fromString("2 :: ,3: "),
            equalTo(Slicer.builder()
                .slice(2, null, null)
                .slice(3, null, null)
                .build()
            )
        );

        assertThat(Slicer.fromString("2: , 3"),
            equalTo(Slicer.builder()
                .slice(2, null)
                .slice(3)
                .build()
            )
        );
    }
}
