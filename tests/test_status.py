import asyncio
import threading

from sheath.status import StatusBus, StatusEvent, bind_bus, current_bus, report_progress


def test_subscriber_receives_emitted_event():
    bus = StatusBus()
    got = []
    bus.subscribe(got.append)
    bus.emit(StatusEvent(tool="t", message="hi", current=1, total=3))
    assert len(got) == 1
    e = got[0]
    assert (e.tool, e.message, e.current, e.total) == ("t", "hi", 1, 3)
    assert e.seq == 1 and e.timestamp > 0          # bus stamps seq + timestamp


def test_seq_is_monotonic_per_bus():
    bus = StatusBus()
    got = []
    bus.subscribe(got.append)
    bus.emit(StatusEvent(tool="t", message="a"))
    bus.emit(StatusEvent(tool="t", message="b"))
    assert [e.seq for e in got] == [1, 2]


def test_report_progress_is_noop_when_unbound():
    assert current_bus() is None
    report_progress("nobody listening")             # must not raise


def test_report_progress_routes_to_bound_bus():
    bus = StatusBus()
    got = []
    bus.subscribe(got.append)
    with bind_bus(bus):
        assert current_bus() is bus
        report_progress("working", current=2, total=4, tool="mytool")
    assert current_bus() is None                     # unbound after the block
    assert (got[0].tool, got[0].message, got[0].current, got[0].total) == ("mytool", "working", 2, 4)


def test_raising_subscriber_does_not_break_emit():
    bus = StatusBus()
    got = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(got.append)                        # second subscriber still runs
    bus.emit(StatusEvent(tool="t", message="ok"))
    assert got and got[0].message == "ok"


def test_unsubscribe_handle_stops_delivery():
    bus = StatusBus()
    got = []
    unsub = bus.subscribe(got.append)
    unsub()
    bus.emit(StatusEvent(tool="t", message="x"))
    assert got == []


def test_emit_from_worker_thread_reaches_subscriber():
    bus = StatusBus()
    got = []
    bus.subscribe(got.append)

    def worker():
        with bind_bus(bus):
            report_progress("from thread", tool="bg")

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert got and got[0].message == "from thread"


def test_bind_bus_enter_and_exit_in_different_tasks_does_not_raise():
    # A continuous Session binds the bus on open (one request's task) and unbinds on close
    # (another task). bind_bus must restore with set(), not reset(token) — otherwise the exit
    # raises "Token was created in a different Context".
    bus = StatusBus()

    async def run():
        holder = {}

        async def enter():
            cm = bind_bus(bus)
            cm.__enter__()              # __enter__ runs in this child task's context
            holder["cm"] = cm

        await asyncio.create_task(enter())
        holder["cm"].__exit__(None, None, None)   # __exit__ in the parent task must not raise

    asyncio.run(run())                  # completes without ValueError
