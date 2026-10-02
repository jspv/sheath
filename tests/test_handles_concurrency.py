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
