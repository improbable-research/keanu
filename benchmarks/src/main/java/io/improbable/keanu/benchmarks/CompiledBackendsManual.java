package io.improbable.keanu.benchmarks;

public class CompiledBackendsManual {

    public static void main(String[] args) {
        CompiledBackEnds compiledBackEnds = new CompiledBackEnds();
        compiledBackEnds.backend = CompiledBackEnds.Backend.KEANU_COMPILED;
        compiledBackEnds.setup();

        for (int i = 0; i < 1; i++) {
            compiledBackEnds.sweepValues();
        }
    }
}
