package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import org.junit.Test;

import static org.mockito.Mockito.mock;

public class AlgorithmFactoryTest {
    @Test
    public void youCanCreateADefaultMetropolisHastingsSampler() {
        KeanuProbabilisticModel model = mock(KeanuProbabilisticModel.class);
        MetropolisHastings mh = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model);
    }

    @Test
    public void youCanCreateACustomMetropolisHastingsSampler() {
        MetropolisHastings.MetropolisHastingsBuilder mh = Keanu.Sampling.MetropolisHastings.builder();

    }

    @Test
    public void youCanCreateADefaultNUTSSampler() {
        NUTS nuts = Keanu.Sampling.NUTS.withDefaultConfig();
    }

    @Test
    public void youCanCreateACustomNUTSSampler() {
        NUTS.NUTSBuilder builder = Keanu.Sampling.NUTS.builder();
    }
}
