package io.improbable.keanu;

import io.improbable.keanu.vertices.VertexId;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

public class DeterministicRule implements TestRule {

    @Override
    public Statement apply(final Statement base, final Description description) {
        KeanuRandom.setDefaultRandomSeed(1);
        VertexId.resetIdGenerator();
        return base;
    }
}
