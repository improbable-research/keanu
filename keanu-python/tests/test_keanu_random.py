from keanu import KeanuRandom

def test_default_keanu_random():
    keanu_random = KeanuRandom()
    random = keanu_random.next_double()

    assert type(random) == float
    assert 0 <= random < 1

def test_seeded_keanu_random():
    keanu_random = KeanuRandom(1)
    random = keanu_random.next_double()

    assert type(random) == float
    assert random == 0.1129943035738381
