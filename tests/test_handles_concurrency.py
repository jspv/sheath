import threading

from tether.handles import HandleStore


def test_concurrent_puts_all_succeed_with_distinct_ids(tmp_path):
    store = HandleStore(tmp_path)
    ids, errors = [], []

    def work():
        for _ in range(50):
            try:
                ids.append(store.put("x" * 100, source="t").id)
            except Exception as e:  # noqa: BLE001 - the test reports any failure
                errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(ids) == 200 and len(set(ids)) == 200
    assert len(HandleStore(tmp_path).manifest()) == 200


def test_readers_do_not_race_concurrent_puts(tmp_path):
    store = HandleStore(tmp_path)
    done = threading.Event()
    errors = []

    def writer():
        try:
            for _ in range(200):
                store.put("x", source="t")
        finally:
            done.set()

    t = threading.Thread(target=writer)
    t.start()
    while not done.is_set():
        try:
            store.manifest()
            store.inputs()
            store.manifest_handles()
            if store._handles:
                store.summary(next(iter(list(store._handles))))
        except Exception as e:  # noqa: BLE001 - the test reports any failure
            errors.append(e)
            break
    t.join()
    assert errors == []
