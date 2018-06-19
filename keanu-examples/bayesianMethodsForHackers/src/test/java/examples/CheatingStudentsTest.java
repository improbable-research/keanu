package examples;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class CheatingStudentsTest {
    @Test
    public void testWhenCheatingStudentsIsRunThenProbabilityOfCheatingIsAccurate() {
        // act
        CheatingStudents.CheatingStudentsPosteriors posteriors = CheatingStudents.run();

        // assert
        assertThat(posteriors.getFreqCheatingMode()).isBetween(0.05, 0.35);
    }
}
