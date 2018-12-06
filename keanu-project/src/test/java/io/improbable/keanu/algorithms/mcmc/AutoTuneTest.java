package io.improbable.keanu.algorithms.mcmc;

import java.util.Collections;

import org.junit.Assert;
import org.junit.Test;

public class AutoTuneTest {

    @Test
    public void canReduceStepsizeFromLargeInitialToSmallToExploreSmallSpace() {
        double startingStepsize = 10.;

        AutoTune tune = new AutoTune(
            Math.log(startingStepsize),
            0.65,
            50
        );

        TreeBuilder treeLessLikely = TreeBuilder.createBasicTree(Collections.EMPTY_MAP, Collections.EMPTY_MAP, Collections.EMPTY_MAP, 0., Collections.EMPTY_MAP);
        treeLessLikely.deltaLikelihoodOfLeapfrog = -50.;
        treeLessLikely.treeSize = 8.;

        double adaptedStepSizeLessLikely = tune.adaptStepSize(treeLessLikely, 1);

        Assert.assertTrue(adaptedStepSizeLessLikely < startingStepsize);

        TreeBuilder treeMoreLikely = TreeBuilder.createBasicTree(Collections.EMPTY_MAP, Collections.EMPTY_MAP, Collections.EMPTY_MAP, 0., Collections.EMPTY_MAP);
        treeMoreLikely.deltaLikelihoodOfLeapfrog = 50.;
        treeMoreLikely.treeSize = 8.;

        double adaptedStepSizeMoreLikely = tune.adaptStepSize(treeMoreLikely, 1);

        Assert.assertTrue(adaptedStepSizeMoreLikely > startingStepsize);
    }

}
